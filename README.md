# 전체 실습 과정을 내 환경에서 정리함. (2020.08.22, MacOS 기준)
- Test code for playing the starcraft2 using reinforcement learning

## STEP 1. Python 환경 설정 (using conda)

### Install miniconda 
- 1) 아래 링크에서 설치파일을 다운로드 받아서 클릭
    - https://docs.conda.io/en/latest/miniconda.html#installing
- 2) bash 명령어로 설치 
    - https://conda.io/projects/conda/en/latest/user-guide/install/macos.html

### 실습에 필요한 python 환경 구성 (virtualenv 생성)
```
cd ~
git clone   
cd ~/dmarl-sc2

conda create -n starcraft2 python=3.6
conda activate starcraft2

conda install pytorch torchvision -c pytorch
pip install tensorflow
pip install pandas
pip install jupyter

pip install ipykernel
python -m ipykernel install --user --name starcraft2 --display-name "starcraft2"

cd ~/dmarl-sc2
jupyter notebook


> conda info --envs
# conda environments:
#
base                  *  /Users/skiper/opt/miniconda3
starcraft2               /Users/skiper/opt/miniconda3/envs/starcraft2
```


## STEP 2. Starcraft 설치 
### pytho 환경 구성
```
cd ~/dmarl-sc2
conda activate starcraft2

pip install --upgrade pip
pip install pysc2
```

### Starcraft 설치 및 게임 맵 다운로드
### 게임 다운로드
- 아래 가이드에 따라 화면을 클릭하여 설치를 완료하면,
- "/Applications/StarCraft\ II/" 디렉토리에 파일이 설치된다 (약 30 GB 사이즈)  
- https://github.com/parksurk/dmarl-sc2/blob/master/PySC2%20StarCraft%20II%20Learning%20Environment%20Setup.md

### 게임 맵 다운로드 (실제 다양한 맵에 따라서 게임 실행)
- 맵을 다운받기 위한 디렉토리 필요. 
```
> cd /Applications/StarCraft\ II/
> mkdir Maps
```
- 아래 링크에서 맵을 다운로드
    - 다운로드 링크 - https://github.com/Blizzard/s2client-proto#map-packs
    - 해당 링크에서 필요한 Map Pack 을 다운로드 받는다 .
    - 이 팩들은 압축을 풀 때 암호를 물어본다.
    - 패스워드 : ‘iagreetotheeula’
    - 아래 실습을 진행할 Simple64 Map은 "Melee"라는 맵을 다운받아야 실행 가능
    ```
    Ladder 2017 Season 1
    Ladder 2017 Season 2
    Ladder 2017 Season 3 Updated
    Ladder 2017 Season 4
    Ladder 2018 Season 1
    Ladder 2018 Season 2
    Ladder 2018 Season 3
    Ladder 2018 Season 4
    Ladder 2019 Season 1
    Ladder 2019 Season 2
    Ladder 2019 Season 3
    Melee
    ```

-  Mini Game 다운로드
    - 다운로드 링크 - https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip

- 다운로드 받은 map list 확인
```
python -m pysc2.bin.map_list
16Bit
    file: 'Ladder2018Season2/16-BitLE.SC2Map'
    battle_net: '16-Bit LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Abiogenesis
    file: 'Ladder2018Season1/AbiogenesisLE.SC2Map'
    battle_net: 'Abiogenesis LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

AbyssalReef
    file: 'Ladder2017Season4/AbyssalReefLE.SC2Map'
    battle_net: 'Abyssal Reef LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

AcidPlant
    file: 'Ladder2018Season3/AcidPlantLE.SC2Map'
    battle_net: 'Acid Plant LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Acolyte
    file: 'Ladder2017Season3/AcolyteLE.SC2Map'
    battle_net: 'Acolyte LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Acropolis
    file: 'Ladder2019Season3/AcropolisLE.SC2Map'
    battle_net: 'Acropolis LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

AscensiontoAiur
    file: 'Ladder2017Season4/AscensiontoAiurLE.SC2Map'
    battle_net: 'Ascension to Aiur LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Automaton
    file: 'Ladder2019Season1/AutomatonLE.SC2Map'
    battle_net: 'Automaton LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Backwater
    file: 'Ladder2018Season1/BackwaterLE.SC2Map'
    battle_net: 'Backwater LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

BattleontheBoardwalk
    file: 'Ladder2017Season4/BattleontheBoardwalkLE.SC2Map'
    battle_net: 'Battle on the Boardwalk LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

BelShirVestige
    file: 'Ladder2017Season1/BelShirVestigeLE.SC2Map'
    battle_net: 'Bel'Shir Vestige LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

BloodBoil
    file: 'Ladder2017Season2/BloodBoilLE.SC2Map'
    battle_net: 'Blood Boil LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Blueshift
    file: 'Ladder2018Season4/BlueshiftLE.SC2Map'
    battle_net: 'Blueshift LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

BuildMarines
    file: 'mini_games/BuildMarines.SC2Map'
    players: 1, score_index: 0, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 0

CactusValley
    file: 'Ladder2017Season1/CactusValleyLE.SC2Map'
    battle_net: 'Cactus Valley LE'
    players: 4, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Catalyst
    file: 'Ladder2018Season2/CatalystLE.SC2Map'
    battle_net: 'Catalyst LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

CeruleanFall
    file: 'Ladder2018Season4/CeruleanFallLE.SC2Map'
    battle_net: 'Cerulean Fall LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

CollectMineralShards
    file: 'mini_games/CollectMineralShards.SC2Map'
    players: 1, score_index: 0, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 0

CollectMineralsAndGas
    file: 'mini_games/CollectMineralsAndGas.SC2Map'
    players: 1, score_index: 0, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 0

CyberForest
    file: 'Ladder2019Season2/CyberForestLE.SC2Map'
    battle_net: 'Cyber Forest LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

DarknessSanctuary
    file: 'Ladder2018Season2/DarknessSanctuaryLE.SC2Map'
    battle_net: 'Darkness Sanctuary LE'
    players: 4, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

DefeatRoaches
    file: 'mini_games/DefeatRoaches.SC2Map'
    players: 1, score_index: 0, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 0

DefeatZerglingsAndBanelings
    file: 'mini_games/DefeatZerglingsAndBanelings.SC2Map'
    players: 1, score_index: 0, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 0

DefendersLanding
    file: 'Ladder2017Season2/DefendersLandingLE.SC2Map'
    battle_net: 'Defender's Landing LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

DiscoBloodbath
    file: 'Ladder2019Season3/DiscoBloodbathLE.SC2Map'
    battle_net: 'Disco Bloodbath LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Dreamcatcher
    file: 'Ladder2018Season3/DreamcatcherLE.SC2Map'
    battle_net: 'Dreamcatcher LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Eastwatch
    file: 'Ladder2018Season1/EastwatchLE.SC2Map'
    battle_net: 'Eastwatch LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Ephemeron
    file: 'Ladder2019Season3/EphemeronLE.SC2Map'
    battle_net: 'Ephemeron LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

FindAndDefeatZerglings
    file: 'mini_games/FindAndDefeatZerglings.SC2Map'
    players: 1, score_index: 0, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 0

Flat128
    file: 'Melee/Flat128.SC2Map'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Flat32
    file: 'Melee/Flat32.SC2Map'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Flat48
    file: 'Melee/Flat48.SC2Map'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Flat64
    file: 'Melee/Flat64.SC2Map'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Flat96
    file: 'Melee/Flat96.SC2Map'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Fracture
    file: 'Ladder2018Season3/FractureLE.SC2Map'
    battle_net: 'Fracture LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Frost
    file: 'Ladder2017Season3/FrostLE.SC2Map'
    battle_net: 'Frost LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Honorgrounds
    file: 'Ladder2017Season1/HonorgroundsLE.SC2Map'
    battle_net: 'Honorgrounds LE'
    players: 4, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Interloper
    file: 'Ladder2017Season3/InterloperLE.SC2Map'
    battle_net: 'Interloper LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

KairosJunction
    file: 'Ladder2019Season2/KairosJunctionLE.SC2Map'
    battle_net: 'Kairos Junction LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

KingsCove
    file: 'Ladder2019Season2/KingsCoveLE.SC2Map'
    battle_net: 'King's Cove LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

LostandFound
    file: 'Ladder2018Season3/LostandFoundLE.SC2Map'
    battle_net: 'Lost and Found LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

MechDepot
    file: 'Ladder2017Season3/MechDepotLE.SC2Map'
    battle_net: 'Mech Depot LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

MoveToBeacon
    file: 'mini_games/MoveToBeacon.SC2Map'
    players: 1, score_index: 0, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 0

NeonVioletSquare
    file: 'Ladder2018Season1/NeonVioletSquareLE.SC2Map'
    battle_net: 'Neon Violet Square LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

NewRepugnancy
    file: 'Ladder2019Season2/NewRepugnancyLE.SC2Map'
    battle_net: 'New Repugnancy LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

NewkirkPrecinct
    file: 'Ladder2017Season1/NewkirkPrecinctTE.SC2Map'
    battle_net: 'Newkirk Precinct TE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Odyssey
    file: 'Ladder2017Season4/OdysseyLE.SC2Map'
    battle_net: 'Odyssey LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

PaladinoTerminal
    file: 'Ladder2017Season1/PaladinoTerminalLE.SC2Map'
    battle_net: 'Paladino Terminal LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

ParaSite
    file: 'Ladder2018Season4/ParaSiteLE.SC2Map'
    battle_net: 'Para Site LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

PortAleksander
    file: 'Ladder2019Season1/PortAleksanderLE.SC2Map'
    battle_net: 'Port Aleksander LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

ProximaStation
    file: 'Ladder2017Season2/ProximaStationLE.SC2Map'
    battle_net: 'Proxima Station LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Redshift
    file: 'Ladder2018Season2/RedshiftLE.SC2Map'
    battle_net: 'Redshift LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Sequencer
    file: 'Ladder2017Season2/SequencerLE.SC2Map'
    battle_net: 'Sequencer LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Simple128
    file: 'Melee/Simple128.SC2Map'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Simple64
    file: 'Melee/Simple64.SC2Map'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Simple96
    file: 'Melee/Simple96.SC2Map'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Stasis
    file: 'Ladder2018Season4/StasisLE.SC2Map'
    battle_net: 'Stasis LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Thunderbird
    file: 'Ladder2019Season3/ThunderbirdLE.SC2Map'
    battle_net: 'Thunderbird LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

Triton
    file: 'Ladder2019Season3/TritonLE.SC2Map'
    battle_net: 'Triton LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

TurboCruise84
    file: 'Ladder2019Season2/TurboCruise84LE.SC2Map'
    battle_net: 'Turbo Cruise '84 LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

WintersGate
    file: 'Ladder2019Season3/WintersGateLE.SC2Map'
    battle_net: 'Winter's Gate LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

WorldofSleepers
    file: 'Ladder2019Season3/WorldofSleepersLE.SC2Map'
    battle_net: 'World of Sleepers LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800

YearZero
    file: 'Ladder2019Season1/YearZeroLE.SC2Map'
    battle_net: 'Year Zero LE'
    players: 2, score_index: -1, score_multiplier: 1
    step_mul: 8, game_steps_per_episode: 28800
```

### 설치 확인 및 