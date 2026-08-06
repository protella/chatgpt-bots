"""THE ONLY PEOPLE ALLOWED TO SPEAK IN THIS REPO'S TESTS, FIXTURES AND PROMPTS.

This repository is PUBLIC. Its scenario tests are reconstructions of real Slack failures, and
every previous round of that reconstruction carried real coworkers' names into a public git
history — twice, the second time only weeks after a history rewrite cleaned up the first.

So the cast is a closed set. Every speaker in a test, every name in a prompt example, every
`who=` / `Say(...)` / roster-map value comes from here and nowhere else. Inventing a name at the
call site is the exact motion that leaked real ones, so `tools/pii_scan.py` fails the build on
any person-shaped name in tests/ that is not in `ROSTER` below — a new character is a one-line
addition here, made deliberately, rather than a name someone typed while writing a scenario.

Names are chosen to be unmistakably fictional and to collide with nobody: no real teammate, no
customer, no vendor contact. If a real person's name ever belongs in this repo it is in LICENSE
and README as the author, and nowhere else.
"""

# The recurring cast the participation scenarios are written around. Five is deliberate: enough
# for a room with an addressee, a bystander and a third party, few enough to keep a reader
# oriented across a hundred scenarios.
ROSTER = (
    "Dana Whitfield",
    "Jamie Jensen",
    "Riley Reyes",
    "Sam Sutton",
    "Tessa Tran",
)

# Walk-on parts: the alphabet cast used where a scenario needs a crowd (member lists, taggable
# rosters, pagination fixtures) and the individuals do not matter.
EXTRAS = (
    "Alice Anderson",
    "Alice Ng",
    "Bob Baker",
    "Carol Chen",
    "David Diaz",
    "Erin Evans",
    "Frank Foster",
    "Grace Green",
    "Henry Hall",
    "Iris Ito",
    "Jack Jones",
    "Karen Kim",
    "Liam Lopez",
    "Maya Mehta",
    "Noah Novak",
    "Olivia Ortiz",
    "Priya Prasad",
    "Quinn Quill",
)

# Bare first names, for the synthetic-name generators and for prompt examples that want one word
# ("Dana, can you take this?"). Every one is the first token of a name above.
FIRST_NAMES = tuple(sorted({n.split()[0] for n in ROSTER + EXTRAS}))

# Non-human speakers. They appear in the same speaker fields and are not person names.
NON_HUMAN = ("ChatGPT", "Claude", "Assistant", "Bot")

ALLOWED_SPEAKERS = frozenset(ROSTER + EXTRAS + NON_HUMAN)
