
The `flush=True` parameter forces the output stream to be written to the console or file immediately instead of being stored in a buffer.
---
It is used to ensure the log message appears in the console immediately for real-time progress tracking rather than being delayed by output buffering, and it does not save space.
---
If `flush=True` is not used, the output may be held in a memory buffer and not appear in the console until the buffer is full or the script finishes, causing a delay in real-time logging.
