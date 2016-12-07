import csv

counter = 0
records = 0
with open("unreliable.csv", "rb") as inp, open("refined.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if (row[3] != "City") and (row[3] != "Town") and (row[3] != "River" and (row[3] != "Bridge") and (row[3] != "Lake") and (row[3] != "Neighborhood") and (row[3] != "Village") and (row[3] != "Trail")  and (row[3] != "Intersection") and (row[3] != "Road")):
             writer.writerow(row)
             records +=1
        else:
            counter += 1

    print counter
    print records