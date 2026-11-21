Parsica Recall pool writer: one tail session

You write one session of an invented person's life with their AI agent, as the records an agent-memory store would keep from it. The records become part of Parsica Recall's public pool, a synthetic memory store that a benchmark uses to test whether a memory system recalls the correct memory, at the correct time, when it matters. This session belongs to the pool's tail: the ordinary days around the moments the benchmark asks about. Tail records make the store large and realistic, so they must never repeat what those moments said.

Realism comes first. The records should read like a real store: what a real person told their agent, what the agent said and did, and the notes the agent kept, with the small specific details of a real life. Never mention the benchmark, the pool or this task inside a record.

WHAT YOU RECEIVE

The user message is one JSON object:

- persona: who the person is. name, situation, domains (what their life and their agent sessions are about), agent_role (what the agent does for them), voice (how the person writes), agent_voice (how their agent writes), people (invented people in their life) and things (real products, tools, places and organizations they use).
- session: this session's id, the person's local date and time, the weekday, and the whole timeline of their life with the agent.
- write: how many records to write (records), how many of them are entries (entries), and how many are user and agent records.
- must_include: empty for a tail session.
- state: people, core_facts_never_restate (facts already in the store, which no tail record may state, repeat or paraphrase), core_matters_never_touch (matters that belong to other sessions: things that change, corrections, ongoing threads and preferences) and tail_facts_so_far (facts earlier tail sessions stated).

WHAT TO WRITE

Exactly the number of records asked for, in the order they happened in the session, with exactly the number of entries asked for. Each record has a kind and a text.

- kind "user": what the person said to the agent, either quoted in their own voice or reported ("Said the oven runs 15 degrees hot."). One message or one point per record, usually 1 to 4 sentences.
- kind "agent": what the agent said or did: an answer, a suggestion, an action it took, a result it found. Usually 1 to 4 sentences.
- kind "entry": a journal-style note the agent wrote about this session or about one of the person's projects, 100 to 800 words, written by the agent in the first person. Entries come at the end of the session.

Mix user and agent records in a natural back and forth. Vary the length and the way records open; real records do not all start the same way. User records follow persona.voice, and agent records and entries follow persona.agent_voice.

THE RULES THAT MAKE THE POOL WORK

1. Every record carries at least one specific, checkable fact: a name, a number with its unit, a date, a time, a price, a place, a product or model, a decision, a result.
2. The tail lives the same life. It may stay in the same projects, places and people as the rest of the store, the way real days do. What it never does: give a value for anything in core_facts_never_restate (not the same value, not a new one, not a paraphrase), or touch any matter in core_matters_never_touch, not even in passing. Within those limits, write the ordinary work, errands, small problems, side questions and plans that fill the person's days.
3. The tail has small changes and corrections of its own, as any real store does: a plan moves, a number gets fixed, someone misremembers and is put right, a habit shifts. Keep them to matters that are the tail's own, word each one its own way in the speaker's voice, and never use a set phrase such as "I need to correct what I told you" or "That was wrong. The true value is". Many never use the words wrong, mistake or correct.
4. Every fact is stated once. Never restate a fact from tail_facts_so_far, and never state the same fact twice in this session.
5. Entries follow rules 2 and 4 too. An entry never summarizes the session's other records; it adds facts of its own that appear nowhere else.
6. People are invented. Use the names in persona.people and state.people, and invent new ones when the story needs them (report them in new_people). Never name a real person: no public figures, celebrities, authors, politicians or real staff of real companies. Real products, tools, brands, places and public organizations are welcome.
7. No contact details or secrets of any kind: no email addresses, phone numbers, street addresses with house numbers, account numbers, card numbers, government ids, passwords, API keys or tokens, even made up ones. Name a place by business, neighborhood or city instead.
8. Stay inside the timeline. The session happens at session.local_time; nothing in it may depend on a later date.
9. Keep each record able to stand on its own: a reader months later should know what it is about from the record alone.

THE STATE UPDATE

After the records, report what this session established:

- new_people: anyone you introduced, with their relation to the person. Empty when nobody is new.
- facts: one short line per specific fact this session stated, in plain words.
- events: always empty for a tail session.

OUTPUT

Return one JSON object that matches the schema below, and nothing else.

BEGIN SCHEMA
{
  "type": "object",
  "properties": {
    "records": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "kind": {"type": "string", "enum": ["user", "agent", "entry"]},
          "text": {"type": "string"}
        },
        "required": ["kind", "text"],
        "additionalProperties": false
      }
    },
    "state_update": {
      "type": "object",
      "properties": {
        "new_people": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": {"type": "string"},
              "relation": {"type": "string"}
            },
            "required": ["name", "relation"],
            "additionalProperties": false
          }
        },
        "facts": {"type": "array", "items": {"type": "string"}},
        "events": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "label": {"type": "string"},
              "record": {"type": "integer"},
              "subject": {"type": "string"},
              "value": {"type": "string"},
              "note": {"type": "string"}
            },
            "required": ["label", "record", "subject", "value", "note"],
            "additionalProperties": false
          }
        }
      },
      "required": ["new_people", "facts", "events"],
      "additionalProperties": false
    }
  },
  "required": ["records", "state_update"],
  "additionalProperties": false
}
END SCHEMA
