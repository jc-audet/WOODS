
import datetime

time = datetime.datetime(2002,1,1,0,0)

train_delta = datetime.timedelta(minutes=30 * (365 * 2*24 * 11 + 3 * 2*24))
val_delta = datetime.timedelta(minutes=30 * (365 * 2*24 * 12 + 3 * 2*24))
test_delta = datetime.timedelta(minutes=30 * (365 * 2*24 * 13 + 3 * 2*24))

print("Train:", time + train_delta, " -- Idx:", 365 * 2*24 * 11 + 3 * 2*24)
print("Val:", time + val_delta, " -- Idx:", 365 * 2*24 * 12 + 3 * 2*24)
print("Test:", time + test_delta, " -- Idx:", 365 * 2*24 * 13 + 3 * 2*24)