---
name: goal-persist
description: 'Persistent goal execution — ensures the AI model NEVER stops until the task is fully completed. Use with /goal command for mission-critical tasks that must reach 100% completion. Triggers on: /goal, goal, mission-critical, do not stop, persist, complete at all costs, never give up, tab tak ruko, kaam khatam karo.'
user-invocable: true
---

# 🎯 Goal Persist — Never Stop Until Done

## Kya Hai Yeh?

Yeh skill ensure karti hai ki AI model **tab tak kaam karta rahe jab tak goal 100% complete na ho jaaye**. Koi bhi excuse nahi — error aaye toh fix karo, rukna nahi, partial answer mat do, poori tarah se khatam karo.

## When to Use

| Trigger | Example |
|---------|---------|
| `/goal <task>` | `/goal Implement full authentication system` |
| `/gaol <task>` | `/gaol Fix all memory leaks` |
| "do not stop until done" | "Do not stop until all tests pass" |
| "persist at all costs" | "Persist at all costs — complete this feature" |
| "tab tak ruko" | "Tab tak ruko jab tak sab kaam na ho jaaye" |
| "kaam khatam karo" | "Isse kaam khatam karo, rukna mat" |

## Procedure (AI Model Follow This Strictly)

### Phase 1: Goal Analysis
1. **Goal ko chhote chhote tasks mein tod do** — ek master todo list banao
2. Har task ko `manage_todo_list` mein daalo with status `not-started`
3. Clear **definition of done** (DoD) har task ke liye likho

### Phase 2: Execution Loop (CRITICAL — Yeh Loop Tab Tak Chale Jab Tak Sab Complete Na Ho)
```
while (any task is not completed) {
    1. Mark ONE task as in-progress
    2. Execute the task (code, research, fix, etc.)
    3. Verify — check errors, test, validate
    4. If FAILED → diagnose, fix, retry (max 3 retries per approach)
    5. If still failing → switch approach, try alternative solution
    6. Mark completed ONLY when DoD is fully met
    7. Move to next task
}
```

### Phase 3: Never-Give-Up Rules
- **Error aaya?** → Ignore karo, fix karo, aage badho. "I can't" bolna allowed nahi hai.
- **Tool failed?** → Alternative tool try karo. Ek tool kaam na kare toh doosra use karo.
- **Context limit aane wala hai?** → Pehle important progress save karo, phir continue karo.
- **Kuch samajh nahi aaya?** → Research karo — web search, files read karo, documentation dekho.
- **Partial solution?** → ALLOWED NAHI HAI. Ya toh poora kaam karo ya phir explicitly user se guidance lo.
- **Thak gaya?** → AI model thakta nahi hai. Bas kaam karte raho.

### Phase 4: Completion Check
- Har task ke baad verify karo ki DoD meet hui ya nahi
- Agar koi task incomplete hai → wapas Phase 2 pe jaao
- Jab saare tasks complete ho jaayein → final summary do

### Phase 5: Final Summary
- Kaunse tasks kiye
- Kaunse files change hue
- Koi known issues / limitations
- Next steps (optional)

## Quality Checklist (AI Model Self-Verify)

- [ ] Har task ka status track kiya?
- [ ] Errors ko handle kiya (ignore nahi kiya)?
- [ ] Partial solution diya? → NO, poora diya?
- [ ] Verification kiya (tests, build, lint)?
- [ ] Final summary diya?

## Edge Cases

| Situation | Action |
|-----------|--------|
| Tool repeatedly fails | Try 3 different approaches, then ask user |
| Missing dependencies | Install them automatically |
| User interrupts | Stop gracefully, save progress |
| Multiple valid solutions | Pick the best one, document why |
| Task is impossible | Explain why with evidence, suggest alternatives |

## Example Usage

```
/goal Add input validation to all API endpoints in the project
```

AI will:
1. Find all API endpoints
2. Break into tasks (one per endpoint)
3. Add validation one by one
4. Test each one
5. Not stop until ALL endpoints are validated