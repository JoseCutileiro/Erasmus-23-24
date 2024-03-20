import Data.List
import System.IO


-- STRUCT (basic example)
data Customer = Customer String String Double 
    deriving Show

lidgi :: Customer 
lidgi = Customer "Luigi" "Pizza" 10.10

getPrice :: Customer -> Double
getPrice (Customer _ _ price) = price


-- STRUCT (rock paper scissors)
data RPS = Rocks | Paper | Scissors

shoot :: RPS -> RPS -> Bool

toText :: Bool -> String

toText result
    | result = "Player 1 wins"
    | otherwise = "Player 1 loses or draws"

shoot Rocks Scissors = True
shoot Scissors Paper  = True
shoot Paper Rocks = True
shoot _ _ = False

main = do 
    print (getPrice(lidgi))

    print (toText(shoot Rocks Rocks))
