import csv

counter = 0
records = 0
with open("original.csv", "rb") as inp, open("maharagama.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if (float(row[1]) < 6.861605) and (float(row[1]) > 6.834293) and (float(row[2]) < 80.217521) and (float(row[2]) > 79.937123):
           writer.writerow(row)
           records += 1
        else:
            counter += 1
    print records

