# Running confused.de on Windows with OpenAI

This is the full path from a clean Windows machine to a working system: install,
configure, verify, run, and use. It assumes no prior setup and no familiarity
with the codebase.

Allow about 20 minutes, most of which is downloads.

---

## What you are installing

A decision console. You describe a decision in plain language; the system
classifies how much human control it requires, answers it from an approved rule
base when one applies, and otherwise holds the answer back for an expert. The
expert's ruling is captured as a rule, so the same judgement is not requested
twice.

Two things are worth knowing before you start.

**The system was built and measured against Anthropic's models.** The OpenAI
backend implements the same six-function interface and passes the same tests,
but the model-quality findings quoted in the paper — in particular that the
cheapest tier could not extract facts reliably — were measured on Claude, not on
GPT. Section 9 says what to watch for.

**Everything except the logic database is disposable.** Conversations live in
process memory and vanish on restart. The rule base is SQLite on disk and
survives. That asymmetry is deliberate.

---

## 1. Install Python

Download Python 3.11 or 3.12 from <https://www.python.org/downloads/windows/>.
Python 3.10 is the minimum; 3.13 works but has fewer pre-built wheels.

In the installer, **tick "Add python.exe to PATH"** on the first screen. This is
the single most common cause of the errors in section 10.

Open **PowerShell** (press `Win`, type `powershell`, press Enter) and confirm:

```powershell
python --version
```

You should see `Python 3.11.x` or similar. If you see a Microsoft Store page or
"not recognized", see section 10.

---

## 2. Get the code

If you have Git:

```powershell
cd $HOME\Documents
git clone <repository-url> ACM_SummerSchool
cd ACM_SummerSchool\prototyping
```

Without Git, download the ZIP, extract it to `Documents`, then:

```powershell
cd $HOME\Documents\ACM_SummerSchool\prototyping
```

Everything below is run from that `prototyping` folder. Confirm you are in the
right place — this should list `server.py`:

```powershell
Get-ChildItem server.py
```

---

## 3. Create a virtual environment

A virtual environment keeps these packages away from the rest of your Python
installation. Skipping it will work today and cause a version conflict later.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`.

If you get **"running scripts is disabled on this system"**, Windows is blocking
the activation script. Allow signed local scripts for your own account only:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Answer `Y`, then run the `Activate.ps1` line again. This affects only your user
account, not the machine.

---

## 4. Install the packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-openai.txt
```

The second file adds the `openai` package. It is kept separate because the
default provider is Anthropic, and an unused SDK is one more thing that can fail
to install on a machine that will never call it.

---

## 5. Get an OpenAI API key

1. Sign in at <https://platform.openai.com/>.
2. Go to **API keys** and choose **Create new secret key**.
3. Copy it immediately — the page will not show it again.
4. Check **Billing**. A key on an account with no credit authenticates
   successfully and then fails on every request with a quota error, which reads
   like a bug in the software. Section 10 covers that message.

A demo session costs a few cents on the default model.

---

## 6. Configure

Copy the example file and edit it:

```powershell
Copy-Item ..\.env.example ..\.env
notepad ..\.env
```

Make it read:

```ini
LLM_PROVIDER="openai"
OPENAI_API_KEY="sk-...your key..."
```

Comment out or delete the Anthropic lines. Save and close.

> **The `.env` file holds a live credential.** It is already listed in
> `.gitignore`, and it must stay that way. Do not paste it into chat, issues, or
> slides. If a key is ever exposed, revoke it on the OpenAI dashboard and issue
> a new one — revocation is immediate and free.

### Choosing models

The page offers three tiers. Their defaults:

| Tier | Default model | Used for |
|---|---|---|
| cheap (default) | `gpt-4o-mini` | conversation, estimating the matrix inputs |
| middle | `gpt-4o` | **floor** for fact extraction and rule drafting |
| strong | `gpt-4.1` | if you select it |

Override any of them in `.env` if your account has different models:

```ini
OPENAI_MODEL_SMALL="gpt-4o-mini"
OPENAI_MODEL_MEDIUM="gpt-4o"
OPENAI_MODEL_LARGE="gpt-5"
```

Two calls per turn — fact extraction and rule drafting — always run at the
middle tier or above, whatever the page is set to. Those calls feed the logic
database, and everything downstream is dead if they return nothing.

### Pointing at something other than OpenAI

The backend speaks Chat Completions, which Azure OpenAI, vLLM, Ollama,
OpenRouter and LM Studio also speak. Add a base URL:

```ini
OPENAI_BASE_URL="http://localhost:11434/v1"
OPENAI_MODEL_SMALL="llama3.1"
```

Local models are untested here. Expect fact extraction to be the first thing
that breaks.

---

## 7. Verify before running

```powershell
python check_setup.py
```

This checks Python, every package, the key, whether each configured model is
actually in your account's catalogue, and then spends about 20 tokens on a real
call — because everything up to that point passes on a key with no credit.

```
  [ ok ] Python 3.10 or newer — found 3.11.9
  [ ok ] package: openai
  [ ok ] API key found — sk-proj…dR4A
  [ ok ] tier haiku   -> gpt-4o-mini  (default)
  [ ok ] model reachable: gpt-4o-mini
  [ ok ] a real call succeeded — 17 in / 4 out
  [ ok ] port 8000 — free
```

Do not continue past a `[FAIL]`. A `[warn]` on model reachability means that
model is not in your account — set the matching override and re-run.

> **Why the tiers are called `haiku`, `sonnet` and `opus` under OpenAI.** Those
> strings are written into stored conversation state, into recorded cases, and
> into the `proposed_by` column of every rule the agent has drafted. Renaming
> them would make an existing database misreport its own history to settle a
> cosmetic complaint. Read them as cheap / middle / strong. The page itself
> never shows them — it asks the server for the real model names.

---

## 8. Start and stop

```powershell
.\start.ps1
```

This launches the server, waits for it to answer, and opens your browser at
<http://127.0.0.1:8000>. Options:

```powershell
.\start.ps1 -Provider openai     # force the provider for this run
.\start.ps1 -Port 8080           # if 8000 is taken
.\start.ps1 -NoBrowser           # do not open a browser
```

To stop:

```powershell
.\stop.ps1
```

Logs are written to `.server.log` and `.server.err` in the same folder. If the
server fails to start, `start.ps1` prints the last 20 lines of both.

The server binds to `127.0.0.1` — your machine only. Nothing on your network can
reach it. There is no authentication, which is correct for a local prototype and
would not be for anything else.

---

## 9. Using it

### The console

Two panels. On the left you chat. On the right you set four properties of the
decision:

| Input | Question it answers |
|---|---|
| **Harm** | how bad is the worst realistic outcome, 0–10 |
| **Reversibility** | how easily could it be undone, 0–10 |
| **Rights-affecting** | does it touch someone's legal or fundamental rights |
| **AI confidence** | how well-evidenced is the assessment, 0–100 |

Each slider's far-right position is **auto**, meaning "let the system estimate
this". Anything you set yourself overrides the estimate. When the system's
estimate is *more* critical than yours, a red bar appears below your slider — it
does not override you, it tells you it disagrees.

The control tier is **computed** from those four, never chosen. Press **?** on
any slider for the definitions the model was given.

### What happens to a question

1. A greeting or a question of fact is answered. Nothing is classified — a
   greeting is not a decision.
2. A described decision is classified, facts are extracted from your wording,
   and the rule base is consulted.
3. If an approved rule covers it, you get the answer with its derivation, and
   nobody waits.
4. If nothing covers it and the tier is consequential, the answer is prepared
   but **held**. You see "waiting for expert approval". You are not shown the
   draft.

### The expert side

Open **Expert review** in the console. Three sections:

- **Inquiries waiting** — each with the answer the agent prepared, in an
  editable box. Change it, then approve. Your edit is what the person receives,
  attributed to you and marked as revised.
- **Rules awaiting review** — clauses drafted from decisions. Approving one
  means future cases matching it are decided without asking anyone.
- **Decisions pulled for audit** — a sample of what was decided automatically.

After approving an answer you are offered the clause behind it. Approve that
too, and the next case of the same shape is decided immediately.

### The logic database

<http://127.0.0.1:8000/database> shows coverage, every rule with its approver
and usage count, every case, and merge candidates. **Reset database** wipes all
of it after a confirmation that names the counts. It cannot be undone.

### A five-minute demonstration

From an empty database, in order:

1. `Should we reorder toner? Stock is low but we have not run out.`
   Low harm, reversible — decided, nobody waits.
2. `We need to decide whether to evict a tenant three months in arrears who
   disputes the debt and was given no formal notice.`
   Human-only — it parks, and nothing is said to you.
3. Open **Expert review**, edit the draft, approve it. Then approve the clause.
4. `Another tenant is in arrears and disputes the amount; no formal notice was
   sent. Do we evict?`
   Decided immediately from the rule you just approved.

Step 4 is the point of the whole system.

> **Do not press "New conversation" between steps 2 and 4.** It clears the
> browser's record of parked cases. The expert still sees them and the answers
> are still stored, but this browser will never show you the reply.

### What differs from the Anthropic build

The measured claim that the cheapest tier extracts facts from 0 of 4 decisions
while the middle tier manages 4 of 4 was made on Claude. The middle-tier floor
is applied to OpenAI as the cautious reading of an unmeasured case, not as a
finding. If coverage on the Logic Database page stays at zero after several
decisions, extraction is returning nothing — raise `OPENAI_MODEL_MEDIUM` to a
stronger model and try again.

---

## 10. When something goes wrong

**`python : The term 'python' is not recognized`**
Python is not on PATH. Re-run the installer, choose **Modify**, and tick "Add
python.exe to PATH". Restart PowerShell afterwards — an open window keeps the
old PATH.

**A Microsoft Store page opens instead of Python**
Windows' app-execution alias is intercepting the name. Settings → Apps →
Advanced app settings → App execution aliases → switch off both `python.exe`
entries.

**`Activate.ps1 cannot be loaded because running scripts is disabled`**
See section 3: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**`RateLimitError: You exceeded your current quota`**
The key is valid; the account has no credit. Add billing on the OpenAI
dashboard. This is not a software fault, despite how it reads.

**`AuthenticationError: Incorrect API key provided`**
Check `.env` for a stray quote, a trailing space, or a truncated paste. Then
confirm it is named exactly `.env` — Notepad will happily save `.env.txt`, and
Explorer hides the extension. Check with `Get-ChildItem ..\.env -Force`. Either
location is read: the repository root (one level above `prototyping`) or
`prototyping` itself; the root is checked first.

**`BadRequestError: ... 'max_tokens' is not supported with this model`**
The backend retries with `max_completion_tokens` automatically, so seeing this
means the retry also failed. Report the model name.

**`model not in this account's catalogue`** from `check_setup.py`
Your account does not have that model. Set the matching `OPENAI_MODEL_*`
override to one it does, and re-run the check.

**Port 8000 already in use**
`.\stop.ps1`, or start elsewhere: `.\start.ps1 -Port 8080`.

**The page loads but the model selector says "default model"**
The page could not read `/api/models`. The server is up but erroring — check
`.server.err`.

**Coverage stays at 0%**
Fact extraction is returning nothing. Raise `OPENAI_MODEL_MEDIUM` and retry. If
it stays at zero with a strong model, the extractor is producing prose instead
of atoms — the first three lines of `.server.log` after a turn will show it.

**Everything worked yesterday and today the rule base is empty**
Conversations are in memory and do not survive a restart, but the rule base is
on disk at `prototyping\data\logic.sqlite`. If it is genuinely empty, either
**Reset database** was pressed, or you are running from a different folder.

---

## 11. Switching back to Anthropic

Edit `..\.env`:

```ini
LLM_PROVIDER="anthropic"
ANTHROPIC_API_KEY="sk-ant-..."
```

then `.\stop.ps1` and `.\start.ps1`. The rule base is provider-independent —
rules approved under one provider are used unchanged by the other. Only the
`proposed_by` column records which model drafted each one.

---

## Appendix — files you will touch

| Path | What it is |
|---|---|
| `..\.env` | your key and provider. Never commit it. |
| `check_setup.py` | pre-flight check. Run it first when anything is wrong. |
| `start.ps1` / `stop.ps1` | run the server |
| `requirements.txt` | base dependencies |
| `requirements-openai.txt` | the `openai` package |
| `agent\backends\openai_api.py` | the OpenAI backend |
| `agent\llm.py` | picks the backend from `LLM_PROVIDER` |
| `data\logic.sqlite` | the rule base. The only thing worth backing up. |
| `.server.log` / `.server.err` | server output |
