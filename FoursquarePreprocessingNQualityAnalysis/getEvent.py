import csv

counter = 0
with open("original.csv", "rb") as inp, open("event.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[4] == "Event":
            writer.writerow(row)
            counter+=1
    print counter