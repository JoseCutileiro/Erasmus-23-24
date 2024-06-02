# @author: Jose Cutileiro
# @group: G70
# Date: 05/05/2024

# This script is needed because Courses.csv has a problem...
import csv
import pandas as pd

programme_mapping = {}
with open('data/Programmes.csv', newline='') as programme_file:
    programme_reader = csv.DictReader(programme_file)
    for row in programme_reader:
        programme_mapping[row['Programme name']] = row['Programme code']


with open('data/Courses.csv', newline='') as courses_file, \
        open('data/CoursesFix.csv', 'w', newline='') as courses_fix_file:
    courses_reader = csv.DictReader(courses_file)
    fieldnames = courses_reader.fieldnames + ['Owned By Id']
    courses_fix_writer = csv.DictWriter(courses_fix_file, fieldnames=fieldnames)
    courses_fix_writer.writeheader()
    
    for course in courses_reader:
        programme_code = course['Owned By']
        programme_name = programme_mapping.get(programme_code, 'Unknown') 
        course['Owned By Id'] = programme_name
        courses_fix_writer.writerow(course)
        
        

data = pd.read_csv('data/Students.csv')

def convert_to_range(year):
    return f"{year}-{year + 1}"

data['Year'] = data['Year'].apply(convert_to_range)

data.to_csv('data/StudentsFix.csv', index=False)