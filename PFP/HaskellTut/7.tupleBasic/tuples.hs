import Data.List
import System.IO


example_tuple = ("Jose", 21)

example_people = ["Jose","Sofia","Leonor"]
example_ages = [21,18,20]

main = do

    -- This is a tuple
    print (1,"Random Tuple")

    -- get first value from tuple
    print (fst example_tuple)

    -- get second value
    print (snd example_tuple)

    -- Combine info
    print (zip example_people example_ages)
