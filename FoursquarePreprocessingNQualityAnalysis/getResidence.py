import csv

counter = 0
with open("original.csv", "rb") as inp, open("residence.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[4] == "Residence":
            writer.writerow(row)
            counter+=1
    print counter