"""Feature compute pipeline.

Stage between the raw event tables (``macro_raw``, ``onchain_raw``, …) and
the model layer (``training_run``, ``signal_history``). Each feature is a
small pure transformation declared in :mod:`.definitions` and applied by
the engine in :mod:`.compute`.

Storage strategy is deliberately *not yet* fixed — the engine returns a
``pandas.DataFrame`` and the worker decides what to do with it (print,
write Parquet, or eventually upsert into a partitioned ``feature_values``
table once the schema is committed to a Flyway migration).
"""
