Please organize this folder as follow:
```
./
├── config.py
├── HMDB51
│   ├── raw
│   │   ├── data -> ../../../../DATA/HMDB51/raw/data
│   │   │   ├── brush_hair
│   │   │   ├── cartwheel
│   │   │   ├── catch
│   │   │   ├── chew
│   │   │   ├── clap
│   │   │   ├── climb
│   │   │   ├── climb_stairs
│   │   │   ├── dive
│   │   │   ├── draw_sword
│   │   │   ├── dribble
│   │   │   ├── drink
│   │   │   ├── eat
│   │   │   ├── fall_floor
│   │   │   ├── fencing
│   │   │   ├── flic_flac
│   │   │   ├── golf
│   │   │   ├── handstand
│   │   │   ├── hit
│   │   │   ├── hug
│   │   │   ├── jump
│   │   │   ├── kick
│   │   │   ├── kick_ball
│   │   │   ├── kiss
│   │   │   ├── laugh
│   │   │   ├── pick
│   │   │   ├── pour
│   │   │   ├── pullup
│   │   │   ├── punch
│   │   │   ├── push
│   │   │   ├── pushup
│   │   │   ├── ride_bike
│   │   │   ├── ride_horse
│   │   │   ├── run
│   │   │   ├── shake_hands
│   │   │   ├── shoot_ball
│   │   │   ├── shoot_bow
│   │   │   ├── shoot_gun
│   │   │   ├── sit
│   │   │   ├── situp
│   │   │   ├── smile
│   │   │   ├── smoke
│   │   │   ├── somersault
│   │   │   ├── stand
│   │   │   ├── swing_baseball
│   │   │   ├── sword
│   │   │   ├── sword_exercise
│   │   │   ├── talk
│   │   │   ├── throw
│   │   │   ├── turn
│   │   │   ├── walk
│   │   │   └── wave
│   │   └── list_cvt
│   │       ├── hmdb51_split1_others.txt
│   │       ├── hmdb51_split1_test.txt
│   │       ├── hmdb51_split1_train.txt
│   │       ├── hmdb51_split2_others.txt
│   │       ├── hmdb51_split2_test.txt
│   │       ├── hmdb51_split2_train.txt
│   │       ├── hmdb51_split3_others.txt
│   │       ├── hmdb51_split3_test.txt
│   │       ├── hmdb51_split3_train.txt
│   │       └── mapping_table.txt
│   └── scripts
│       ├── convert_list.py
│       └── resave_videos.py
├── __init__.py
├── Kinetics
│   ├── raw
│   │   ├── data -> ../../../../DATA/Kinetics/raw/data
│   │   │   ├── test
│   │   │   ├── train
│   │   │   └── val
│   │   └── list_cvt
│   │       ├── kinetics_test.txt
│   │       ├── kinetics_test_avi.txt
│   │       ├── kinetics_train.txt
│   │       ├── kinetics_train_avi.txt
│   │       ├── kinetics_val.txt
│   │       ├── kinetics_val_avi.txt
│   │       └── mapping_table.txt
│   └── scripts
│       ├── convert_video.py
│       └── remove_spaces.py
├── README.md
└── UCF101
    └── raw
        ├── data -> ../../../../DATA/UCF101/raw/data
        │   ├── ApplyEyeMakeup
        │   ├── ApplyLipstick
        │   ├── Archery
        │   ├── BabyCrawling
        │   ├── BalanceBeam
        │   ├── BandMarching
        │   ├── BaseballPitch
        │   ├── Basketball
        │   ├── BasketballDunk
        │   ├── BenchPress
        │   ├── Biking
        │   ├── Billiards
        │   ├── BlowDryHair
        │   ├── BlowingCandles
        │   ├── BodyWeightSquats
        │   ├── Bowling
        │   ├── BoxingPunchingBag
        │   ├── BoxingSpeedBag
        │   ├── BreastStroke
        │   ├── BrushingTeeth
        │   ├── CleanAndJerk
        │   ├── CliffDiving
        │   ├── CricketBowling
        │   ├── CricketShot
        │   ├── CuttingInKitchen
        │   ├── Diving
        │   ├── Drumming
        │   ├── Fencing
        │   ├── FieldHockeyPenalty
        │   ├── FloorGymnastics
        │   ├── FrisbeeCatch
        │   ├── FrontCrawl
        │   ├── GolfSwing
        │   ├── Haircut
        │   ├── Hammering
        │   ├── HammerThrow
        │   ├── HandstandPushups
        │   ├── HandstandWalking
        │   ├── HeadMassage
        │   ├── HighJump
        │   ├── HorseRace
        │   ├── HorseRiding
        │   ├── HulaHoop
        │   ├── IceDancing
        │   ├── JavelinThrow
        │   ├── JugglingBalls
        │   ├── JumpingJack
        │   ├── JumpRope
        │   ├── Kayaking
        │   ├── Knitting
        │   ├── LongJump
        │   ├── Lunges
        │   ├── MilitaryParade
        │   ├── Mixing
        │   ├── MoppingFloor
        │   ├── Nunchucks
        │   ├── ParallelBars
        │   ├── PizzaTossing
        │   ├── PlayingCello
        │   ├── PlayingDaf
        │   ├── PlayingDhol
        │   ├── PlayingFlute
        │   ├── PlayingGuitar
        │   ├── PlayingPiano
        │   ├── PlayingSitar
        │   ├── PlayingTabla
        │   ├── PlayingViolin
        │   ├── PoleVault
        │   ├── PommelHorse
        │   ├── PullUps
        │   ├── Punch
        │   ├── PushUps
        │   ├── Rafting
        │   ├── RockClimbingIndoor
        │   ├── RopeClimbing
        │   ├── Rowing
        │   ├── SalsaSpin
        │   ├── ShavingBeard
        │   ├── Shotput
        │   ├── SkateBoarding
        │   ├── Skiing
        │   ├── Skijet
        │   ├── SkyDiving
        │   ├── SoccerJuggling
        │   ├── SoccerPenalty
        │   ├── StillRings
        │   ├── SumoWrestling
        │   ├── Surfing
        │   ├── Swing
        │   ├── TableTennisShot
        │   ├── TaiChi
        │   ├── TennisSwing
        │   ├── ThrowDiscus
        │   ├── TrampolineJumping
        │   ├── Typing
        │   ├── UnevenBars
        │   ├── VolleyballSpiking
        │   ├── WalkingWithDog
        │   ├── WallPushups
        │   ├── WritingOnBoard
        │   └── YoYo
        └── list_cvt
            ├── testlist01.txt
            ├── testlist02.txt
            ├── testlist03.txt
            ├── trainlist01.txt
            ├── trainlist02.txt
            └── trainlist03.txt
```
