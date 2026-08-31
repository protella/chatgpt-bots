<!--
  WORKSPACE CONTEXT — optional but recommended.

  This file gives the bot durable background about YOUR organization: what the company does,
  what the products are, and what your acronyms mean. It is injected verbatim into every
  system prompt, so the bot reads it on every message in every channel and DM.

  SETUP
    1. Copy this file:  cp workspace_context.example.md workspace_context.md
    2. Point .env at it: WORKSPACE_CONTEXT_FILE=workspace_context.md
    3. Restart the bot. Edits are picked up on restart, not live.
    4. Keep your real file OUT of version control if the repo is public.

  WHAT BELONGS HERE
    Durable facts that are true next month: the business, the product line, the vocabulary,
    who the audience is. This is the difference between the bot guessing what "SNAP staging"
    means and it actually knowing.

  WHAT DOES NOT BELONG HERE
    - Anything that changes week to week (sprint status, who is on call, current deploys).
      The bot cannot see live systems; stale facts here become confident wrong answers.
    - Secrets, credentials, or customer data.
    - Behavior instructions ("always answer in bullet points"). This file is background
      knowledge, not a rules file — per-user custom instructions and channel memory are
      where behavior belongs.

  Keep it tight. A page or two of dense, real information beats ten pages of filler.
  Delete these comments and the placeholder content below, then write your own.
-->

# About Example Corp

Example Corp is a market-research company serving the restaurant and packaged-goods industries.
We sell data and insights to product developers, marketers, and strategists who need to know what
consumers eat, what restaurants are putting on menus, and where a category is heading.

Our customers are mostly enterprise: chain restaurant brands, food manufacturers, and the
agencies that serve them.

# Products

- **Atlas** — flagship menu-trend database. Tracks items, ingredients, and pricing across
  national and regional chains. Customers use it for competitive tracking and concept ideation.
- **Pulse** — consumer survey platform. Recurring panels on eating habits, flavor preferences,
  and purchase drivers.
- **Beacon** — the reporting and dashboard layer that sits on top of Atlas and Pulse. This is
  what most customers actually log into.
- **Foundry** — internal data pipeline that ingests and normalizes menu data. Not customer-facing.

# Acronyms and internal terms

| Term | Meaning |
|------|---------|
| MTD | Menu Trend Database — the data behind Atlas |
| FSR | Full-Service Restaurant |
| QSR | Quick-Service Restaurant |
| LTO | Limited-Time Offer — a seasonal or promotional menu item |
| CPG | Consumer Packaged Goods |
| NPD | New Product Development |
| "the panel" | Pulse's recurring consumer survey respondents |
| "staging" | Our pre-production environment, where releases are verified before going live |

# How we work

- Engineering runs two-week sprints. Release tickets are tracked in Jira under the PS project.
- Environments are `dev` → `staging` → `production`. A release is announced in the team channel
  when it happens, but not every deploy is posted — absence of a message is not evidence a
  deploy did not run.
- The company operates primarily US Central time.
