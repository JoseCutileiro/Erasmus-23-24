# Parallel functional programming

# Aula 1

## Lei de Moore

"O número de transistors por chip dúplica a cada X anos" - O número de X aumenta exponencialmente

Um computador tem três partes:
1. CPU 
2. Armazenamento 
3. Conexão entre o CPU e o armazenamento

Esta conexão é vista como o bottleneck

Mais frequência de relogio -> Mais energia é consumida

O futuro é paralelo -> Somar em vez de multiplicar 

## Parallel programming is hard? 

1. Race condictions: Acontecimentos em simultaneo levam a incorreções, comportamento não deterministico. É uma chatice fazer debug disto ...
2. Locks são propicios a erros, por falhar ou esquecimento
3. Locks podem ter problemas acrescidos levando a deadlocks (ou outras coisas)
4. Fazer um lock é caro - Vale mais ou menos 30 writes

## Programação funcional

1. Os dados são imutaveis - Podem ser partilhados sem problemas
2. Computações paralelas não têm side-effects

## Compilar parallel haskell

1. Adicionar main function (exemplo)

```
main = print(nfib 40)
```

2. Compilar

```
ghc -O2 -threaded -rtsopts -eventlog -feager-blackholing NF.hs
```

3. Run the code

```
NF.exe

NF.exe +RTS -N1 (usar um core só)

NF.exe +RTS -N4 -ls (usar quatro cores e um event log)
```

!!! SLIDE 22 !!!