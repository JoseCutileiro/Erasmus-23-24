import Data.List
import System.IO

-- IFS
doubleEvenNumber y = 
    if (mod y 2 /= 0)
        then y
        else y  * 2

-- CASE
getClass n =  case n of
    5 -> "Kindergarten"
    12 -> "High school"
    _ -> "Noob"

main = do 
    
    print (doubleEvenNumber 2)
    print (doubleEvenNumber 3)

    print (getClass 5)
    print (getClass 12)
    print (getClass 13)


