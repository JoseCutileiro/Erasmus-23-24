-- Selecionar particular elementares
SELECT ?particle ?particleLabel
WHERE {
  ?particle wdt:P279 wd:Q43116
            
   SERVICE wikibase:label {
     bd:serviceParam wikibase:language "en"  
   }
}

-- Selectionar 10 cantoras (filtrar por genero)
SELECT ?singer ?singerLabel
WHERE {
  ?singer wdt:P106 wd:Q177220.
  ?singer wdt:P21 wd:Q6581072.
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en".
  }
} LIMIT 10


-- Selecionas 10 cantoras (com imagens)
SELECT ?singer ?singerLabel ?singerImage
WHERE {
  ?singer wdt:P106 wd:Q177220.
  ?singer wdt:P21 wd:Q6581072.
  
  ?singer wdt:P18 ?singerImage .

  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en".
  }
} LIMIT 10


-- Da mesma maneira que fomos buscar a imagem podemos ir buscar qualquer informação relacionada com 
-- singer. E para além disso, quando vamos buscar a imagem podemos ir buscar informação associada à imagem também
-- (é um processo de procura de informação recursivo)