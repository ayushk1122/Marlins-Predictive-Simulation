import pybaseball as pb
data = pb.statcast('2024-01-01', '2024-01-02')
print("All Statcast columns:", list(data.columns)) 