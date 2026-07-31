"""One identity for this process, minted before anything can need it.

`SESSION_ID` used to live inside participation_telemetry, behind that module's enable flag.
Outbound receipts (spec §5) key dead-session reconciliation on the session half of every
`turn_id`, so the value has to exist whether or not the ledger is switched on — and it has to
be the SAME value, or a ledger line and a receipt row from one run would name two sessions.
"""

import uuid

SESSION_ID = uuid.uuid4().hex
