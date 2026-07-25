# Golden eval set

JSONL, one question per line. `golden.jsonl`.

## Fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | string | yes | `q001`, stable across edits |
| `question` | string | yes | Phrased how someone would actually ask it |
| `source_docs` | string[] | yes | Filenames that contain the answer. Used for recall@k. |
| `answer` | string | yes | Short reference answer, for faithfulness scoring |
| `answer_spans` | string[] | no | Verbatim phrases that must appear in the retrieved text |
| `difficulty` | enum | yes | `easy`, `medium`, `hard` |
| `failure_mode` | string | no | What this question is designed to break |

## Difficulty means something specific here

- **easy**. answer sits in one section. question uses the same words the doc uses
- **medium**. answer is in one doc but the question uses different vocabulary
- **hard**. answer needs two or more docs, or the obvious keyword match is the wrong doc

Hard questions are the point. A retriever that only handles `easy` scores well and helps
nobody.

## Validating

```bash
python -m evalset.validate
```

Checks the schema, that ids are unique, and that every `source_docs` entry exists in the
fetched corpus. Run it after fetching.

## Growing the set

Twenty questions is thin. Target is 60 by day 5, added when a retrieval failure is found
that no existing question covers. Adding questions the current system already passes is
how eval sets stop being useful.
