import csv

counter = 0
records = 0
with open("original.csv", "rb") as inp, open("awissawellaC.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if (float(row[1]) < 6.977122) and (float(row[1]) > 6.931572) and (float(row[2]) < 80.217521) and (float(row[2]) > 80.184562):
           writer.writerow(row)
           records += 1
        else:
            counter += 1
    print records

