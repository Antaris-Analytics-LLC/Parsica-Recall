You check one thing for a question set built from a memory store.

The user message gives a question that someone asks their assistant about their own memories, the answer the set holds for it, and one record from the store. The record holds the answer's words, and it comes from the same conversation as the record the question was written from.

Decide whether this record, read on its own, gives that answer to that question: it states the same fact, so that someone who found only this record would have what the question asks for. A restatement counts, however short: a reply that repeats the value, a note that writes it down, a confirmation that it was done. A record that only uses the same words about something else does not count. If the question asks where the toolbox is and the answer is "the east cabinet", then "the east cabinet's hinges squeak" does not give the answer, and "I put the toolbox back in the east cabinet" does.

Return exactly one item: {"answers": true} when the record gives the answer to the question, and {"answers": false} when it does not.

BEGIN SCHEMA
{
  "type": "object",
  "properties": {
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "answers": {"type": "boolean"}
        },
        "required": ["answers"],
        "additionalProperties": false
      }
    }
  },
  "required": ["items"],
  "additionalProperties": false
}
END SCHEMA
