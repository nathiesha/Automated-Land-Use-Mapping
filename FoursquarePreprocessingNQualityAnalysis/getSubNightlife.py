import csv

cat = set()
counter = 0
with open("original.csv", "rb") as inp, open("subNightlife.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[4] == "Nightlife Spot":
            if row[3] not in cat:
                writer.writerow(row)
                cat.add(row[3])
    print cat.__len__()
    print cat
