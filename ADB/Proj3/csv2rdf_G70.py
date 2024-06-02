# @author: Jose Cutileiro
# @group: G70
# Date: 05/05/2024

import csv

DATA_PATH = "data/"
OUTPUT_FILE = "out.ttl"

def reset_file(file_path):
    try:
        with open(file_path, 'w'):
            pass
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    

def loadStudents():
    with open(DATA_PATH + "Students.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                student_id = row["Student id"]
                student_name = row["Student name"]
                graduated = row["Graduated"]
                output_file.write(f":{student_id} rdf:type :Student  .\n")
                output_file.write(f":{student_id} :studentName \"{student_name}\"  .\n")
                output_file.write(f":{student_id} :studentId \"{student_id}\"  .\n")
                output_file.write(f":{student_id} :graduated \"{graduated}\"  .\n")
                output_file.write("\n")
           
def loadTeachers():
    with open(DATA_PATH + "Teaching_Assistants.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                teacher_id = row["Teacher id"]
                teacher_name = row["Teacher name"]
                output_file.write(f":{teacher_id} rdf:type :Teacher  .\n")
                output_file.write(f":{teacher_id} :teacherName \"{teacher_name}\"  .\n")
                output_file.write(f":{teacher_id} :teacherId \"{teacher_id}\"  .\n")
                output_file.write("\n")

    with open(DATA_PATH + "Senior_Teachers.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                teacher_id = row["Teacher id"]
                teacher_name = row["Teacher name"]
                output_file.write(f":{teacher_id} rdf:type :Teacher  .\n")
                output_file.write(f":{teacher_id} rdf:type :Senior_Teacher  .\n")
                output_file.write(f":{teacher_id} :teacherName \"{teacher_name}\"  .\n")
                output_file.write(f":{teacher_id} :teacherId \"{teacher_id}\"  .\n")
                output_file.write("\n")

def loadProgrammes():
    with open(DATA_PATH + "Programmes.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                programme_code = row["Programme code"]
                programme_name = row["Programme name"]
                output_file.write(f":{programme_code} rdf:type :Programme  .\n")
                output_file.write(f":{programme_code} :programmeName \"{programme_name}\"  .\n")
                output_file.write(f":{programme_code} :programmeCode \"{programme_code}\"  .\n")
                output_file.write("\n")

def loadCourses():
    with open(DATA_PATH + "Courses.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                level = row["Level"]
                course_name = row["Course name"]
                course_code = row["Course code"]
                credits_ = row["Credits"]
                
                output_file.write(f":{course_code} rdf:type :Course  .\n")
                output_file.write(f":{course_code} :courseName \"{course_name}\"  .\n")
                output_file.write(f":{course_code} :courseCode \"{course_code}\"  .\n")
                output_file.write(f":{course_code} :credits \"{credits_}\"  .\n")
                output_file.write(f":{course_code} :level \"{level}\"  .\n")
                output_file.write("\n")


def relationship_OWNS():
    with open(DATA_PATH + "CoursesFix.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                owned_by = row["Owned By Id"]
                course_code = row["Course code"]
                output_file.write(f":{owned_by} :owns :{course_code}  .\n")
                output_file.write("\n")

def loadCourseInstance():
    with open(DATA_PATH + "Course_Instances.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                academic_year = row["Academic year"]
                _id = row["Instance_id"]
                course_code = row["Course code"]
                study_period = row["Study period"]
                output_file.write(f":{_id} rdf:type :Course_Instance  .\n")
                output_file.write(f":{_id} :academicYearCourse \"{academic_year}\"  .\n")
                output_file.write(f":{_id} :instanceId \"{_id}\"  .\n")
                output_file.write(f":{_id} :studyPeriod \"{study_period}\"  .\n")
                output_file.write(f":{course_code} :hasExecution :{_id}  .\n")
                output_file.write("\n")

    with open(DATA_PATH + "Course_plannings.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                num_students = row["Planned number of Students"]
                senior_hours = row["Senior Hours"]
                assistant_hours = row["Assistant Hours"]
                _id = row["Course"]
                output_file.write(f":{_id} :assistantHours \"{assistant_hours}\"  .\n")
                output_file.write(f":{_id} :seniorHours \"{senior_hours}\"  .\n")
                output_file.write(f":{_id} :planningNumStudents \"{num_students}\"  .\n")
                output_file.write("\n")

def relationship_EXAMINES():
    with open(DATA_PATH + "Course_Instances.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                examiner = row["Examiner"]
                instance_id = row["Instance_id"]
                output_file.write(f":{examiner} :examine :{instance_id}  .\n")
                output_file.write("\n")

def loadDept():
    with open(DATA_PATH + "Teaching_Assistants.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                _id = row["Department name"]
                teacher_id = row["Teacher id"]
                output_file.write(f":{_id} rdf:type :Dept  .\n")
                output_file.write(f":{_id} :deptName \"{_id}\"  .\n")
                output_file.write(f":{teacher_id} :teacherDept :{_id}  .\n")

    with open(DATA_PATH + "Senior_Teachers.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                _id = row["Department name"]
                teacher_id = row["Teacher id"]
                output_file.write(f":{_id} rdf:type :Dept  .\n")
                output_file.write(f":{_id} :deptName \"{_id}\"  .\n")
                output_file.write(f":{teacher_id} :teacherDept :{_id}  .\n")

    with open(DATA_PATH + "Courses.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                _id = row["Department"]
                key_course = row["Course code"]
                output_file.write(f":{_id} rdf:type :Dept  .\n")
                output_file.write(f":{_id} :deptName \"{_id}\"  .\n")
                output_file.write(f":{key_course} :courseDept :{_id}  .\n")

    with open(DATA_PATH + "Programmes.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                _id = row["Department name"]
                key_programme = row["Programme code"]
                output_file.write(f":{_id} rdf:type :Dept  .\n")
                output_file.write(f":{_id} :deptName \"{_id}\"  .\n")
                output_file.write(f":{key_programme} :programmeDept :{_id}  .\n")

def loadDivision():
    with open(DATA_PATH + "Teaching_Assistants.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                _id = row["Division name"]
                dept_name = row["Department name"]
                teacher_id = row["Teacher id"]
                output_file.write(f":{_id} rdf:type :Division  .\n")
                output_file.write(f":{_id} :divisionName \"{_id}\"  .\n")
                output_file.write(f":{teacher_id} :teacherDivision :{_id}  .\n")
                output_file.write(f":{_id} :hasDept :{dept_name}  .\n")
                

    with open(DATA_PATH + "Senior_Teachers.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                _id = row["Division name"]
                dept_name = row["Department name"]
                teacher_id = row["Teacher id"]
                output_file.write(f":{_id} rdf:type :Division  .\n")
                output_file.write(f":{_id} :divisionName \"{_id}\"  .\n")
                output_file.write(f":{teacher_id} :teacherDivision :{_id}  .\n")
                output_file.write(f":{_id} :hasDept :{dept_name}  .\n")

    with open(DATA_PATH + "Courses.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                _id = row["Division"]
                dept_name = row["Department"]
                key_course = row["Course code"]
                output_file.write(f":{_id} rdf:type :Division  .\n")
                output_file.write(f":{_id} :divisionName \"{_id}\"  .\n")
                output_file.write(f":{key_course} :courseDivision :{_id}  .\n")
                output_file.write(f":{_id} :hasDept :{dept_name}  .\n")

def loadRegistrations():
    with open(DATA_PATH + "Registrations.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                # Using the key as name for the variable
                key = row["Course Instance"] + row["Student id"]
                course_key = row["Course Instance"]
                student_key = row["Student id"]
                grade = row["Grade"]
                status = row["Status"]
                output_file.write(f":{key} rdf:type :Registration  .\n")
                output_file.write(f":{key} :status \"{status}\"  .\n")
                output_file.write(f":{key} :grade \"{grade}\"  .\n")
                output_file.write(f":{course_key} :hasRegistration :{key}  .\n")
                output_file.write(f":{student_key} :register :{key}  .\n")
                output_file.write("\n")

def loadHours():
    with open(DATA_PATH + "Reported_Hours.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                # Using the key as name for the variable
                key = row["Course code"] + row["Teacher Id"]
                assigned_hours = row["Hours"]
                output_file.write(f":{key} rdf:type :Hour_Information  .\n")
                output_file.write(f":{key} :reportedHours \"{assigned_hours}\"  .\n")
                output_file.write("\n")

    with open(DATA_PATH + "Assigned_Hours.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                # Using the key as name for the variable
                key = row["Course Instance"] + row["Teacher Id"]
                assigned_hours = row["Hours"]
                output_file.write(f":{key} rdf:type :Hour_Information  .\n")
                output_file.write(f":{key} :assignedHours\"{assigned_hours}\"  .\n")
                output_file.write("\n")

def loadProgrammeCourses():
    with open(DATA_PATH + "Programme_Courses.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                # Using the key as name for the variable
                academic_year = row["Academic Year"]
                study_year = row["Study Year"]
                course_type = row["Course Type"]
                course_id = row["Course"]
                programme_code = row["Programme code"]
                key = programme_code + academic_year
                output_file.write(f":{key} rdf:type :Programme_Course  .\n")
                output_file.write(f":{key} :academicYearProgramme \"{academic_year}\"  .\n")
                output_file.write(f":{key} :courseType \"{course_type}\"  .\n")
                output_file.write(f":{key} :studyYear \"{study_year}\"  .\n")
                output_file.write(f":{key} :belongsTo :{programme_code}  .\n")
                output_file.write(f":{key} :includedIn :{course_id}  .\n")
                output_file.write("\n")

def relationship_HOURS():
    # ASSOCIATED WITH AND GIVEN 
    with open(DATA_PATH + "Reported_Hours.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                # Using the key as name for the variable
                key_hours = row["Course code"] + row["Teacher Id"]
                key_teacher = row["Teacher Id"]
                key_course_instance = row["Course code"]
                output_file.write(f":{key_hours} :associatedWith :{key_teacher}  .\n")
                output_file.write(f":{key_hours} :given :{key_course_instance}  .\n")
                output_file.write(f":{key_teacher} :worksIn :{key_course_instance}  .\n")
                output_file.write("\n")    

    with open(DATA_PATH + "Assigned_Hours.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                # Using the key as name for the variable
                key_hours = row["Course Instance"] + row["Teacher Id"]
                key_teacher = row["Teacher Id"]
                key_course_instance = row["Course Instance"]
                
                output_file.write(f":{key_hours} :associatedWith :{key_teacher}   .\n")
                output_file.write(f":{key_hours} :given :{key_course_instance}  .\n")
                output_file.write(f":{key_teacher} :worksIn :{key_course_instance}  .\n")
                output_file.write("\n")
 

def relationship_OVERSEES(): 
    with open(DATA_PATH + "Programmes.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                # Using the key as name for the variable
                key_programme = row["Programme code"]
                key_director = row["Director"]
                output_file.write(f":{key_director} :oversees :{key_programme}  .\n")
                output_file.write("\n") 

def relationship_ENROLLS():
    with open(DATA_PATH + "StudentsFix.csv", 'r') as input_file:
        reader = csv.DictReader(input_file)
        with open(OUTPUT_FILE, 'a') as output_file:
            for row in reader:
                student_id = row["Student id"]
                programme_code = row["Programme"]
                academic_year = row["Year"]
                key = programme_code + academic_year
                output_file.write(f":{student_id} :enrolls :{key}  .\n")
                output_file.write("\n")

# RESET FILE
reset_file(OUTPUT_FILE)

# LOAD BASIC ATRIBUTES
loadStudents()
loadTeachers()
loadProgrammes()
loadCourses()
loadCourseInstance()               # AND RELATIONSHIPS
loadDept()                         # AND RELATIONSHIPS
loadDivision()                     # AND RELATIONSHIPS
loadRegistrations()                # AND RELATIONSHIPS
loadHours()
loadProgrammeCourses()

# LOAD RELATIONS
relationship_OWNS()
relationship_EXAMINES()
relationship_HOURS()               # AND WORKS IN
relationship_OVERSEES()
relationship_ENROLLS()