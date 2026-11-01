"""
Plain speaker notes for the long presentation, one entry per slide in slide order.
Used by scripts/build_long_deck.py (slide notes) and scripts/speaker_notes_pdf.py (printable page).

shows: what is on the screen, in one line
say:   what to say, a few short sentences
asked: a likely question and the short answer
"""

NOTES = [
    {"title": "Driver Aggressiveness Index",
     "shows": "title",
     "say": "This week: I checked the labels by hand, added new signals, let a network learn the formula, and tested the "
            "model on German roads."},

    {"title": "Pipeline",
     "shows": "the five steps from raw data to a label",
     "say": "Data goes in, gets cleaned, we compute signals, the formula turns them into a score, and the agent decides "
            "conservative, normal or aggressive. The three dark boxes are what changed this week.",
     "asked": "What does cleaning do? Missing values stay missing instead of becoming 0, and impossible values are removed."},

    {"title": "Equations",
     "shows": "every formula used in one table",
     "say": "Top row is the reference formula: a weighted average of the terms we know. Extended adds three new terms. "
            "KAN learns the shape of each term from data. The last row is the new reward: waiting costs more when the "
            "driver is in danger.",
     "asked": "What is d? Danger, from 0 to 1: it grows as time to collision drops under 4 s or the "
              "headway drops under 1 s."},

    {"title": "Formula against label on UAH",
     "shows": "2x2 table: label vs formula, and a doughnut",
     "say": "Most windows agree (1,903 and 1,243). Two groups disagree. 325 windows: the label says aggressive but the "
            "formula says calm; UAH's own software sides with the formula 72% of the time, so the label is wrong there. "
            "697 windows: the label says normal and the formula says aggressive; I watched the videos and these drivers "
            "are not aggressive, so the formula is wrong there.",
     "asked": "Why is the label wrong? UAH labels the whole trip, so calm moments in an aggressive trip are still "
              "labelled aggressive."},

    {"title": "Two sampled windows",
     "shows": "one example from each disagreeing group, signal by signal",
     "say": "Left: the gap closes to 10 m at the end. The label is right and the formula reacts too late, which is why "
            "TTC was added. Right: 100 km/h in a 120 zone. The formula calls it aggressive because it only sees raw "
            "speed, which is why speed versus the limit was added."},

    {"title": "The same two windows on video",
     "shows": "the road video of those two windows",
     "say": "Same two windows on video. Left is closing in on the car ahead. Right is ordinary motorway driving behind "
            "trucks."},

    {"title": "From a hard cut-off to a gradual probability",
     "shows": "grey: the hard cut-off; maroon: what the data really shows",
     "say": "The grey line is our rule: below 27.5 normal, above it aggressive. One point can flip the label. The maroon "
            "line is the real share of aggressive trips at each score: it rises slowly. 27% of windows sit within 2 "
            "points of a cut-off, so near identical drivers get opposite labels. That is the problem to fix.",
     "asked": "How do you fix it? Three ways, next slide."},

    {"title": "What smoothing does to the disagreements",
     "shows": "the three smoothing methods compared with the hard cut-off",
     "say": "Probability: say '60% aggressive' instead of a hard label. Borderline band: scores between 25 and 30 are "
            "'not sure yet', the agent keeps watching. Hysteresis: only switch to aggressive after two windows in a row "
            "above 30, like a thermostat. Probability removes the most false alarms (697 down to 382). The band has the "
            "best accuracy (81%) on the windows it decides. Hysteresis is the only one that also catches some of the "
            "missed aggressive windows.",
     "asked": "What is the catch with the band? About a quarter of windows wait for more data instead of getting a "
              "label right away."},

    {"title": "What each signal is worth on its own",
     "shows": "AUC of each signal alone, darker is better",
     "say": "AUC: 0.5 is a coin flip, 1 is perfect. The physics signal (IDM residual: how far the driver is from what a "
            "normal car-following model would do) is one of the best on both real and simulated drivers. Time to "
            "collision alone is a coin flip, because dangerous moments are short, so it went into the reward instead.",
     "asked": "Why is IDM so high on SUMO? SUMO cars are IDM cars, so it is partly by construction."},

    {"title": "Reference, extended formula and KAN on each UAH driver",
     "shows": "radar: one spoke per driver, further out is better",
     "say": "Each driver is tested after training on the other five. The reference scores 0.842, the extended formula "
            "0.860 and the KAN 0.894. The KAN is best on all six drivers."},

    {"title": "Shapes the KAN learned from the data",
     "shows": "the curve the KAN learned for each input",
     "say": "Instead of us choosing squared or linear, the KAN learns the shape. It learned acceleration straight, not "
            "squared, and speed curved, mattering mostly at high speed. That is the opposite of the change we made last "
            "week, so it is a question for Dr. Daher.",
     "asked": "What is a KAN? A network where each input goes through its own learned curve and the curves are added, "
              "so every curve can be plotted and read."},

    {"title": "Accuracy against training length",
     "shows": "accuracy of PPO and DQN as training gets longer",
     "say": "One million steps takes 2.6 minutes. PPO is flat between 71 and 74% from the start. DQN explores randomly "
            "at first, then catches up. Training longer does not help: the limit is the information the agent gets, not "
            "the training time."},

    {"title": "Only new information moves the ceiling",
     "shows": "waterfall: what each change added to accuracy",
     "say": "Start at 72.2%. New signals add 7.3 points. The risk reward and three times longer training add almost "
            "nothing. We end at 79.8% on the simulated test drivers."},

    {"title": "Three experts switched by a state machine",
     "shows": "one expert per road type vs one shared agent",
     "say": "A small state machine picks the expert for the current road type. Experts win on urban and weather and are "
            "equal overall (80.1% vs 79.7%). The experts use the same rule with thresholds shifted per road type, so one "
            "agent with a threshold per road type does the same job."},

    {"title": "Simulated German autobahn",
     "shows": "animation of the simulated autobahn, the traffic mix and the rules",
     "say": "Three lanes, 120 km/h then no speed limit. Cars keep right, only aggressive ones overtake on the right, and "
            "15% are trucks at 80 km/h. Colours: blue conservative, gold normal, red aggressive, grey trucks."},

    {"title": "Transfer to the autobahn",
     "shows": "four agents tested on German runs they never saw",
     "say": "A, trained on other roads with the old inputs: 72.6%. B, same roads but with relative speeds and the physics "
            "signal: 80.8%, without ever seeing a German road. Adding German data and a country flag (C, D) did not "
            "help yet, but the German training set is small and those were single runs.",
     "asked": "Why do relative speeds help? 130 km/h means something different on an autobahn than in the US; speed "
              "relative to the limit and to the traffic means the same everywhere."},

    {"title": "Real highD, rule-violation labels",
     "shows": "real German highway data, labelled by breaking a rule",
     "say": "highD has no aggressive labels, so a window is marked aggressive when the driver follows closer than 0.9 s "
            "or drives 20% over the limit: 26% of windows. With all terms the score is 0.97, but that is circular. "
            "Without gap and limit, the extended formula still finds the violators at 0.83."},

    {"title": "Replay: how the agent decides",
     "shows": "second by second replay of two drivers",
     "say": "Left: the agent follows a car, sees enough and labels normal, correct. Right: the driver is never close to "
            "a car, so the agent has little to go on and labels conservative, which is wrong."},

    {"title": "RAG: where it plugs in",
     "shows": "where retrieval would feed the agent",
     "say": "The RAG was built in October but is not inside any model yet. Plan: find similar normal windows (same road "
            "type, traffic, limit, country) and give the agent how this driver ranks among them. The rule files are only "
            "for explaining a decision."},

    {"title": "What comes next",
     "shows": "six next steps",
     "say": "Smooth the label with the band and hysteresis. Decide on the KAN shapes with Dr. Daher. Put the RAG context "
            "in the agent. Repeat the German test with more seeds. Run full DriveDNA once access is granted. Use the "
            "separate parts of the index to decide when to intervene."},
]
