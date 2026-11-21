Parsica Recall pool writer: one core session

You write one session of an invented person's life with their AI agent, as the records an agent-memory store would keep from it. The records become part of Parsica Recall's public pool, a synthetic memory store that a benchmark uses to test whether a memory system recalls the correct memory, at the correct time, when it matters. Later, other people write questions whose answers sit in these records, and a memory system has to find the right record among thousands.

Realism comes first. The records should read like a real store: what a real person told their agent, what the agent said and did, and the notes the agent kept, with the small specific details of a real life. Never mention the benchmark, the pool or this task inside a record.

WHAT YOU RECEIVE

The user message is one JSON object:

- persona: who the person is. name, situation, domains (what their life and their agent sessions are about), agent_role (what the agent does for them), voice (how the person writes), agent_voice (how their agent writes), people (invented people in their life) and things (real products, tools, places and organizations they use).
- session: this session's id, the person's local date and time, the weekday, and the whole timeline of their life with the agent.
- write: how many records to write (records), how many of them are entries (entries), and how many are user and agent records.
- must_include: things this session has to contain, each with a label such as V2, C1 or T3. Each one goes in its own record.
- state: what earlier sessions established. people, facts_so_far (one line per fact already stated, with its date), open_items (the labeled items started earlier, with their subject and values) and never_mention_again (facts that were said once and must never come up again).

WHAT TO WRITE

Exactly the number of records asked for, in the order they happened in the session, with exactly the number of entries asked for. Each record has a kind and a text.

- kind "user": what the person said to the agent, either quoted in their own voice or reported ("Said the oven runs 15 degrees hot."). One message or one point per record, usually 1 to 4 sentences.
- kind "agent": what the agent said or did: an answer, a suggestion, an action it took, a result it found. Usually 1 to 4 sentences.
- kind "entry": a journal-style note the agent wrote about this session or about one of the person's projects, 100 to 800 words, written by the agent in the first person. Entries come at the end of the session.

Mix user and agent records in a natural back and forth. Vary the length and the way records open; real records do not all start the same way. User records follow persona.voice, and agent records and entries follow persona.agent_voice.

THE RULES THAT MAKE THE POOL WORK

1. Every record carries at least one specific, checkable fact: a name, a number with its unit, a date, a time, a price, a place, a product or model, a decision, a result. "Sounds good" is not a record; "Booked the 7:40 ferry to Bainbridge for Saturday" is.
2. Every fact is stated once in the whole store. Never restate a fact from facts_so_far, open_items or never_mention_again, and never state the same fact twice in this session. A later record may refer to something earlier in general words ("the ferry plan", "that repair") but must not repeat its details.
3. Entries follow rule 2 too. An entry never summarizes the session's other records. It adds facts of its own that appear nowhere else: what the agent noticed, decided, planned or measured, the status of a project, a number it worked out, a follow-up it set, context it looked up. Write it the way a careful assistant keeps project notes, not as a recap.
4. Each must_include item goes in its own record, as the item says, and in no other record. Follow its instructions exactly, including who speaks and the way it is put: a change states the new value without repeating the old one, a correction names the earlier value and gives the right one, a continued topic adds a new detail without repeating earlier ones, and a shown preference is shown through a choice or action and never stated.
5. Changes and corrections sound like real people, never like a form. Each one is worded its own way, in the speaker's voice, the way the item describes. No set phrases such as "I need to correct what I told you" or "That was wrong. The true value is", no opening with a date, and many corrections never use the words wrong, mistake or correct at all: a person often just says what it actually is.
6. People are invented. Use the names in persona.people and state.people, and invent new ones when the story needs them (report them in new_people). Never name a real person: no public figures, celebrities, authors, politicians or real staff of real companies. Real products, tools, brands, places and public organizations are welcome.
7. No contact details or secrets of any kind: no email addresses, phone numbers, street addresses with house numbers, account numbers, card numbers, government ids, passwords, API keys or tokens, even made up ones. Name a place by business, neighborhood or city instead.
8. Stay inside the timeline. The session happens at session.local_time; nothing in it may depend on a later date, and nothing may contradict state.
9. Keep each record able to stand on its own: a reader months later should know what it is about from the record alone, without the records around it.

THE STATE UPDATE

After the records, report what this session established, so later sessions stay consistent:

- new_people: anyone you introduced, with their relation to the person. Empty when nobody is new.
- facts: one short line per specific fact this session stated, in plain words. These lines are never shown to anyone except later sessions of this writer.
- events: one per must_include item, with its label, its record (the position of the record that carries it in your records list, counting from 0), its subject (what it is about, in a few words), its value (the value, preference, detail or wrong statement given in this session) and a note (for a correction's first part, the true value; otherwise usually empty).

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
