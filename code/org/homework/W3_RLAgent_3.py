import random
import time
import math
import os.path

import numpy as np
import pandas as pd


from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

DATA_FILE = 'rlagent_learning_data'

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack' # 64 * 64 pixel이므로 4096개의 좌표로 공격이 가능

ACTION_BUILD_REFINARY = "buildrefinary"

# Actions for zerg
ACTION_SELECT_LARVA = 'selectlarva'
ACTION_SELECT_DRONE = 'selectdrone'
ACTION_BUILD_OVERLOAD = 'buildoverload'
ACTION_BUILD_SPAWNING_POOL = 'buildspawningpool'
ACTION_BUILD_ZERGLING = 'buildzergling'
ACTION_BUILD_HATCHERY = 'buildhatchery'
ACTION_BUILD_DRONE = 'builddrone'



# 아래 action 중 ACTION_ATTACK은 하나의 위치로 정할수 없기 때문에
# 4096개의 ACTION_ATTCK을 생성해야 함. (실습을 위해서는 16개로만 지정)
smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_SELECT_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_SELECT_ARMY,
]

smart_actions_zerg = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_LARVA,
    ACTION_SELECT_DRONE,
    ACTION_BUILD_OVERLOAD,
    ACTION_BUILD_SPAWNING_POOL,
    ACTION_SELECT_ARMY,
    ACTION_BUILD_HATCHERY,
    ACTION_BUILD_ZERGLING,
    ACTION_BUILD_DRONE
]

#for mm_x in range(0, 64):
#    for mm_y in range(0, 64):
#        smart_actions.append(ACTION_ATTACK + '_' + str(mm_x) + '_' + str(mm_y))

# ACTION_ATTACK을 smart_actions에 추가
for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 16 == 0 and (mm_y + 1) % 16 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))
            smart_actions_zerg.append(ACTION_ATTACK + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))

# Reward를 제공할 가중치 (빌딩 파괴시 더 큰 보상)
KILL_UNIT_REWARD = 0.6
KILL_BUILDING_REWARD = 0.3


# reference from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        print(self.q_table.columns, len(self.q_table.columns))
        print(len(self.actions))
        print("Test")

    # q-table에서 가장 값이 높은 action을 선택한다
    # 이때 e_greedy의 확률만큼만 다른 액선을 랜덤으로 선택한다.
    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            # state_action = self.q_table.ix[observation, :]
            state_action = self.q_table.loc[observation, :]

            # some actions have the same value (가장 높은 값의 액션이 여러개 있는 경우 랜덤으로 1개만)
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        # q_predict = self.q_table.ix[s, a] (현재 상태 Q-Value 값)
        q_predict = self.q_table.loc[s, a]
        # q_target = r + self.gamma * self.q_table.ix[s_, :].max() (다음 상태 예측 Q-Value 값)
        q_target = r + self.gamma * self.q_table.loc[s_, :].max()

        # update (게임 중 새로운 상태가 발생하면 계속 q-table에 상태를 추가 )
        # self.q_table.ix[s, a] += self.lr * (q_target - q_predict)
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            print(self.actions)
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class TerranRLAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TerranRLAgent, self).__init__()

        self.base_top_left = None
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0

        self.previous_action = None
        self.previous_state = None

        # if os.path.isfile(DATA_FILE + '.gz'):
        #     self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]

        return [x, y]

    def getMeanLocation(self, unitList):
        sum_x = 0
        sum_y = 0
        for unit in unitList:
            sum_x += unit.x
            sum_y += unit.y
        mean_x = sum_x / len(unitList)
        mean_y = sum_y / len(unitList)

        return [mean_x, mean_y]

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
            obs.observation.single_select[0].unit_type == unit_type):
              return True

        if (len(obs.observation.multi_select) > 0 and
            obs.observation.multi_select[0].unit_type == unit_type):
              return True

        return False

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def step(self, obs):
        super(TerranRLAgent, self).step(obs)

        #time.sleep(0.5)

        if obs.last():
            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        supply_depot_count = len(self.get_units_by_type(obs, units.Terran.SupplyDepot))

        barracks_count = len(self.get_units_by_type(obs, units.Terran.Barracks))

        supply_limit = obs.observation.player.food_cap
        army_supply = obs.observation.player.food_used

        killed_unit_score = obs.observation.score_cumulative.killed_value_units
        killed_building_score = obs.observation.score_cumulative.killed_value_structures

#        current_state = np.zeros(5000)
#        current_state[0] = supply_depot_count
#        current_state[1] = barracks_count
#        current_state[2] = supply_limit
#        current_state[3] = army_supply
#
        # 적군의 위치를 파악한다. (4096 pixel을 찾아서 적이 위치한 곳을 파악)
        # 4096은 너무 시간이 걸리므로, 아래의 코드로 16개 포인트로 적용
#        hot_squares = np.zeros(4096)
#        enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.ENEMY).nonzero()
#        for i in range(0, len(enemy_y)):
#            y = int(enemy_y[i])
#            x = int(enemy_x[i])
#
#            hot_squares[((y - 1) * 64) + (x - 1)] = 1
#
#        if not self.base_top_left:
#            hot_squares = hot_squares[::-1]
#
#        for i in range(0, 4096):
#            current_state[i + 4] = hot_squares[i]

        current_state = np.zeros(21)
        current_state[0] = supply_depot_count
        current_state[1] = barracks_count
        current_state[2] = supply_limit
        current_state[3] = army_supply

        # 적군의 위치를 16개 포인트로 지정하여 업데이트 하는 코드 
        hot_squares = np.zeros(16)
        enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.ENEMY).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 16))
            x = int(math.ceil((enemy_x[i] + 1) / 16))

            hot_squares[((y - 1) * 4) + (x - 1)] = 1

        if not self.base_top_left:
            hot_squares = hot_squares[::-1]

        for i in range(0, 16):
            current_state[i + 4] = hot_squares[i]

        if self.previous_action is not None:
            reward = 0
            # Reward를 제공하는 함수 (상태 유닛이 kill된 개수에 따라 보상)
            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD
            # 파괴한 상대 건물의 개수만큼 보상
            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD

            # Q-table을 업데이트 하는 코드 (위에서 계산된 보상을 기반으로)
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]

        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_state = current_state
        self.previous_action = rl_action

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        if smart_action == ACTION_DO_NOTHING:
            return actions.FUNCTIONS.no_op()

        elif smart_action == ACTION_SELECT_SCV:
            if self.can_do(obs, actions.FUNCTIONS.select_point.id):
                scvs = self.get_units_by_type(obs, units.Terran.SCV)
                if len(scvs) > 0:
                    scv = random.choice(scvs)
                    if scv.x >= 0 and scv.y >= 0:
                        return actions.FUNCTIONS.select_point("select", (scv.x,
                                                                         scv.y))

        # Build_Refinery_screen
        elif smart_action == ACTION_BUILD_REFINARY:
            if self.can_do(obs, actions.FUNCTIONS.Build_Refinery_screen.id):
                ccs = self.get_units_by_type(obs, units.Terran.CommandCenter)
                if len(ccs) > 0:
                    mean_x, mean_y = self.getMeanLocation(ccs)
                    target = self.transfotransformDistancermDistance(int(mean_x), 0, int(mean_y), 20)

                    return actions.FUNCTIONS.Build_SupplyDepot_screen("now", target)
        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            if self.can_do(obs, actions.FUNCTIONS.Build_SupplyDepot_screen.id):
                ccs = self.get_units_by_type(obs, units.Terran.CommandCenter)
                if len(ccs) > 0:
                    mean_x, mean_y = self.getMeanLocation(ccs)
                    target = self.transformDistance(int(mean_x), 0, int(mean_y), 20)

                    return actions.FUNCTIONS.Build_SupplyDepot_screen("now", target)

        elif smart_action == ACTION_BUILD_BARRACKS:
            if self.can_do(obs, actions.FUNCTIONS.Build_Barracks_screen.id):
                ccs = self.get_units_by_type(obs, units.Terran.CommandCenter)
                if len(ccs) > 0:
                    mean_x, mean_y = self.getMeanLocation(ccs)
                    target = self.transformDistance(int(mean_x), 20, int(mean_y), 0)

                    return actions.FUNCTIONS.Build_Barracks_screen("now", target)

        elif smart_action == ACTION_SELECT_BARRACKS:
            if self.can_do(obs, actions.FUNCTIONS.select_point.id):
                barracks = self.get_units_by_type(obs, units.Terran.Barracks)
                if len(barracks) > 0:
                    barrack = random.choice(barracks)
                    if barrack.x >= 0 and barrack.y >= 0:
                        return actions.FUNCTIONS.select_point("select", (barrack.x,
                                                                              barrack.y))

        elif smart_action == ACTION_BUILD_MARINE:
            if self.can_do(obs, actions.FUNCTIONS.Train_Marine_quick.id):
                return actions.FUNCTIONS.Train_Marine_quick("queued")

        elif smart_action == ACTION_SELECT_ARMY:
            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")

        elif smart_action == ACTION_ATTACK:
            #if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
            if not self.unit_type_is_selected(obs, units.Terran.SCV) and self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                return actions.FUNCTIONS.Attack_minimap("now", self.transformLocation(int(x), int(y)))

        return actions.FUNCTIONS.no_op()



class ZergRLAgent(base_agent.BaseAgent):
    def __init__(self):
        super(ZergRLAgent, self).__init__()

        # !!!!!
        self.attack_coordinates = None

        self.base_top_left = None
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions_zerg))))

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0

        self.previous_action = None
        self.previous_state = None

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')


    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]

        return [x, y]

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def step(self, obs):
        super(ZergRLAgent, self).step(obs)

        # time.sleep(0.5)
        if obs.last():
            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()
            # xmean = player_x.mean()
            # ymean = player_y.mean()
            #
            # if xmean <= 31 and ymean <= 31:
            #     self.attack_coordinates = (49, 49)
            # else:
            #     self.attack_coordinates = (12, 16)
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        # 매 STEP마다 이전 Action의 결과 (overload개수, zergling 개수 등)를 업데이트 (보상을 위한 단계)
        overload_count = len(self.get_units_by_type(obs, units.Zerg.Overlord))
        zergling_count = len(self.get_units_by_type(obs, units.Zerg.Zergling))

        supply_limit = obs.observation.player.food_cap
        army_supply = obs.observation.player.food_used

        killed_unit_score = obs.observation.score_cumulative.killed_value_units
        killed_building_score = obs.observation.score_cumulative.killed_value_structures
        saved_minerals_score = obs.observation.player.minerals
        # print(saved_minerals_score)
        current_state = np.zeros(21)
        current_state[0] = overload_count
        current_state[1] = zergling_count
        current_state[2] = supply_limit
        current_state[3] = army_supply

        # 적군의 위치를 16개 포인트로 지정하여 업데이트 하는 코드
        hot_squares = np.zeros(16)
        enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.ENEMY).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 16))
            x = int(math.ceil((enemy_x[i] + 1) / 16))

            hot_squares[((y - 1) * 4) + (x - 1)] = 1

        if not self.base_top_left:
            hot_squares = hot_squares[::-1]

        # 위에서는 총 21개를 current_state에 할당했는데,
        # hot_square(적군의 위치)를 16개 합해도 기존 4개 + 16개 = 20개만 할당됨.
        # 나머지 1개는 무슨 용도인가?
        for i in range(0, 16):
            current_state[i + 4] = hot_squares[i]

        if self.previous_action is not None:
            reward = 0
            # Reward를 제공하는 함수 (상태 유닛이 kill된 개수에 따라 보상)
            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD
            # 파괴한 상대 건물의 개수만큼 보상
            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD
            if reward > 0: print("Reward 1: ", reward)


            if zergling_count > self.previous_zergling_count:
                reward += 0.2
                if supply_limit > self.previous_supply_limit:
                    reward *= 1.2
                if army_supply > self.previous_army_count:
                    reward *= 1.2
            if saved_minerals_score > 2000:
                reward *= 0.7

            if reward > 0: print("Reward 2: ", reward)
            # Q-table을 업데이트 하는 코드 (위에서 계산된 보상을 기반으로)
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions_zerg[rl_action]

        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_state = current_state
        self.previous_action = rl_action
        self.previous_supply_limit = supply_limit
        self.previous_army_count = army_supply
        self.previous_zergling_count = zergling_count

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        # print(smart_action)

        # ACTION_SELECT_ZERGLING,
        # ACTION_SELECT_ARMY,

        # ACTION_ATTACK
        # ACTION_BUILD_SPAWNING_POOL,
        # ACTION_SELECT_LARVA,
        # ACTION_SELECT_DRONE,
        # ACTION_BUILD_OVERLOAD,

        if smart_action == ACTION_DO_NOTHING:
            return actions.FUNCTIONS.no_op()
        elif smart_action == ACTION_SELECT_LARVA:
            larvae = self.get_units_by_type(obs, units.Zerg.Larva)
            if len(larvae) > 0:
                larva = random.choice(larvae)
                return actions.FUNCTIONS.select_point("select_all_type", (larva.x,larva.y))
        elif smart_action == ACTION_SELECT_DRONE:
            drones = self.get_units_by_type(obs, units.Zerg.Drone)
            if len(drones) > 0:
                drone = random.choice(drones)
                return actions.FUNCTIONS.select_point("select_all_type", (drone.x,drone.y))
        # ATTACK
        elif smart_action == ACTION_ATTACK:
            if self.unit_type_is_selected(obs, units.Zerg.Zergling):
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    zerglings = self.get_units_by_type(obs, units.Zerg.Zergling)
                    return actions.FUNCTIONS.Attack_minimap("now", self.transformLocation(int(x), int(y)))
        elif smart_action == ACTION_SELECT_ARMY:
            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")
        # ACTION_BUILD_SPAWNING_POOL
        elif smart_action == ACTION_BUILD_SPAWNING_POOL:
            spawning_pools = self.get_units_by_type(obs, units.Zerg.SpawningPool)
            if len(spawning_pools) == 0:
                if self.unit_type_is_selected(obs, units.Zerg.Drone):
                    if self.can_do(obs, actions.FUNCTIONS.Build_SpawningPool_screen.id):
                        x = random.randint(0, 83)
                        y = random.randint(0, 83)

                        return actions.FUNCTIONS.Build_SpawningPool_screen("now", (x, y))

            # drones = self.get_units_by_type(obs, units.Zerg.Drone)
            # if len(drones) > 0:
            #     drone = random.choice(drones)
            #     return actions.FUNCTIONS.select_point("select_all_type", (drone.x,
            #                                                               drone.y))
        # ACTION_BUILD_OVERLOAD
        elif smart_action == ACTION_BUILD_OVERLOAD:
            if self.unit_type_is_selected(obs, units.Zerg.Larva):
                free_supply = (obs.observation.player.food_cap -
                               obs.observation.player.food_used)
                # if free_supply < 4:
                if self.can_do(obs, actions.FUNCTIONS.Train_Overlord_quick.id):
                    return actions.FUNCTIONS.Train_Overlord_quick("now")
        # ACTION_BUILD_ZERGLING
        elif smart_action == ACTION_BUILD_ZERGLING:
                if self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
                    return actions.FUNCTIONS.Train_Zergling_quick("now")
        # ACTION_BUILD_DRONE
        elif smart_action == ACTION_BUILD_DRONE:
            drones = self.get_units_by_type(obs, units.Zerg.Drone)
            # if len(drones) < 10:
            if self.can_do(obs, actions.FUNCTIONS.Train_Drone_quick.id):
                return actions.FUNCTIONS.Train_Drone_quick("now")
        elif smart_action == ACTION_BUILD_HATCHERY:
            hatcher_count = self.get_units_by_type(obs, units.Zerg.Hatchery)
            if len(hatcher_count) == 0 or len(hatcher_count) < 5:
                # if(saved_minerals_score > 1000):
                if self.unit_type_is_selected(obs, units.Zerg.Drone):
                    if self.can_do(obs, actions.FUNCTIONS.Build_Hatchery_screen.id):
                        x = random.randint(0, 83)
                        y = random.randint(0, 83)

                        return actions.FUNCTIONS.Build_Hatchery_screen("now", (x, y))
#########################################################################################
        # zerglings = self.get_units_by_type(obs, units.Zerg.Zergling)
        # if len(zerglings) >= 10:
        #     if self.unit_type_is_selected(obs, units.Zerg.Zergling):
        #         if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
        #             return actions.FUNCTIONS.Attack_minimap("now",
        #                                                     self.attack_coordinates)
        #
        #     if self.can_do(obs, actions.FUNCTIONS.select_army.id):
        #         return actions.FUNCTIONS.select_army("select")
        #
        # spawning_pools = self.get_units_by_type(obs, units.Zerg.SpawningPool)
        # if len(spawning_pools) == 0:
        #     if self.unit_type_is_selected(obs, units.Zerg.Drone):
        #         if self.can_do(obs, actions.FUNCTIONS.Build_SpawningPool_screen.id):
        #             x = random.randint(0, 83)
        #             y = random.randint(0, 83)
        #
        #             return actions.FUNCTIONS.Build_SpawningPool_screen("now", (x, y))
        #
        #     drones = self.get_units_by_type(obs, units.Zerg.Drone)
        #     if len(drones) > 0:
        #         drone = random.choice(drones)
        #
        #         return actions.FUNCTIONS.select_point("select_all_type", (drone.x,
        #                                                                   drone.y))
        # if self.unit_type_is_selected(obs, units.Zerg.Larva):
        #     free_supply = (obs.observation.player.food_cap -
        #                    obs.observation.player.food_used)
        #     if free_supply == 0:
        #         if self.can_do(obs, actions.FUNCTIONS.Train_Overlord_quick.id):
        #             return actions.FUNCTIONS.Train_Overlord_quick("now")
        #
        #     if self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
        #         return actions.FUNCTIONS.Train_Zergling_quick("now")
        #
        # larvae = self.get_units_by_type(obs, units.Zerg.Larva)
        # if len(larvae) > 0:
        #     larva = random.choice(larvae)
        #
        #     return actions.FUNCTIONS.select_point("select_all_type", (larva.x,
        #                                                               larva.y))



        return actions.FUNCTIONS.no_op()

def main(unused_argv):
    # agent = TerranRLAgent()
    agent = ZergRLAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    #map_name="AbyssalReef",
                    map_name="Simple64",
                    players=[sc2_env.Agent(sc2_env.Race.zerg),
                    # players=[sc2_env.Agent(sc2_env.Race.terran),
                             sc2_env.Bot(sc2_env.Race.random,
                                         sc2_env.Difficulty.very_easy)],
                    agent_interface_format=features.AgentInterfaceFormat(
                      feature_dimensions=features.Dimensions(screen=84, minimap=64),
                      use_feature_units=True),
                    step_mul=8,
                    game_steps_per_episode=0,
                    visualize=True) as env:

              agent.setup(env.observation_spec(), env.action_spec())

              timesteps = env.reset()
              agent.reset()

              while True:
                  step_actions = [agent.step(timesteps[0])]
                  if timesteps[0].last():
                      break
                  timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)
