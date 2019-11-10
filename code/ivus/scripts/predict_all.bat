FOR /l %%j IN (0, 1, 4) DO (
    python predict.py 5 %%j
)