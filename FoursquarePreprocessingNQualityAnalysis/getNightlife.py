import csv

counter = 0
with open("original.csv", "rb") as inp, open("nightlife.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[4] == "Nightlife Spot":
            writer.writerow(row)
            counter+=1
    print counter