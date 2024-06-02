{-# LANGUAGE FlexibleInstances, MultiParamTypeClasses,UndecidableInstances, FlexibleContexts, TypeSynonymInstances #-}
import Data.Monoid
import Control.Applicative
import Control.Monad
import Control.Monad.Writer
import Control.Monad.Trans
import System.IO
import Data.IORef



instance Functor (C m)  where
   fmap = liftM

instance Applicative (C m)  where
   pure  = return
   (<*>) = ap

data Action m = 
       Atom (m (Action m))           -- do some work
     | Fork (Action m) (Action m)    -- create a new thread
     | Stop                          -- finish the thread


newtype C m a = C { apply :: (a -> Action m) -> Action m }
  

instance Monad (C m) where

  f >>= k  = C $ \ c -> apply f ( \ v -> apply (k v) c ) 

  return x = C $ \ c -> c x


atom :: Monad m => m a -> C m a 
atom f = C $ \ c -> Atom $ do { v <- f; return (c v) }


instance MonadTrans C where
  -- lift :: Monad m => m a -> C m a
  lift = atom


stop :: Monad m => C m a 
stop = C $ \c -> Stop

par :: Monad m => C m a -> C m a -> C m a
par f1 f2 = C $ \c -> Fork (apply f1 c) (apply f2 c)

fork :: Monad m => C m () -> C m ()
fork f = C $ \c -> Fork (action f) (c ())

action :: Monad m => C m a -> Action m
action f = apply f (\v -> Stop)

sched :: Monad m => [Action m] -> m ()
sched [] = return ()
sched (a : as) = case a of 
  Atom am    -> do { a' <- am ; sched (as ++ [a']) }
  Fork a1 a2 -> sched (as ++ [a1,a2])
  Stop       -> sched as

run :: Monad m => C m a -> m ()
run m = sched  [ action m ]



class Monad m => Output m where  
   write :: String -> m ()


instance Output IO where
   write = putStr


loop :: Output m => String -> m ()
loop s = do write s 
            loop s

-- instance Output m => Output (C m) where
--     write s = lift (write s)


example0 :: Output m => C m ()
example0 = do write "start!"
              par  (loop "yellow ") (loop "cat")

example1 :: Output m => C m ()
example1 = do write "start!"
              fork  (loop "black")
	      loop "cat"




-- Two different possibilities for how to define write (string-wise or char-wise)


-- instance Output m => Output (C m) where
--     write s = lift (write s)


instance Output m => Output (C m) where
  write [] = lift (write [])
  write (x:xs) = lift (write [x]) >> write xs





