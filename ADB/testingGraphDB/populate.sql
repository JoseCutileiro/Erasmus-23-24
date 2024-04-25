-- Populate
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX : <http://www.semanticweb.org/35196/ontologies/2024/3/test/>
INSERT DATA {
    :Jose rdf:type :Boy .
    :Jose :name "José Cutileiro" .
    
    :Leonor rdf:type :Girl .
    :Leonor :name "Leonor Cardeal" .
}

-- First query example
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX : <http://www.semanticweb.org/35196/ontologies/2024/3/test/>
select ?p ?n where {
    ?p rdf:type :Person .
    ?p :name ?n
} limit 100

-- output: 
-- Jose && Leonor