SHELL = /bin/bash

BOLD := \033[1m
DIM := \033[2m
RED := \033[31m
RESET := \033[0m

DATA_DIR = ./data/
WIKIDATA_MAPPINGS_DIR = ${DATA_DIR}wikidata-mappings/
TYPE_FEATURES_DIR = ${DATA_DIR}computed-mappings/
PREDICATE_VARIANCES_DIR = ${DATA_DIR}predicate-variances/
RESULTS_DIR = ${DATA_DIR}results/
TRIPLES_FILE = ${RESULTS_DIR}natural_types.ttl

WIKIDATA_SPARQL_ENDPOINT = https://qlever.cs.uni-freiburg.de/api/wikidata
# Note that the query names are also used for generating the file name by
# casting the query name to lowercase and appending .tsv
DATA_QUERY_NAMES = QID_TO_DESCRIPTION QID_TO_LABEL QID_TO_SITELINKS QID_TO_P31 QID_TO_P279

# If you want to use all features, including those that need to be precomputed, run: make triples FEATURES=all MODEL=models/nn.512_sig_d04_64_adam00001.70k.pt
FEATURES = no_precomputed_features  # Alternatively: all
MODEL = models/nn.no_precomp.512_sigmoid_d02_32_adam00001.70k.pt  # Alternatively: models/nn.512_sig_d04_64_adam00001.70k.pt

triples:
	@[ -d ${RESULTS_DIR} ] || mkdir ${RESULTS_DIR}
	@if ls ${TRIPLES_FILE} 1> /dev/null 2>&1; then \
	  echo -e "$${RED}Natural entity type triples file already exists at ${TRIPLES_FILE} . Delete (e.g. by running \"make delete-triples\") or rename it if you want to create a new triples file.$${RESET}"; echo; \
	else \
	  python3 scripts/create_natural_type_triples.py --load_model ${MODEL} -o ${TRIPLES_FILE} -m 12 --feature_set ${FEATURES}; \
	fi

delete-triples:
	@if ls ${TRIPLES_FILE} 1> /dev/null 2>&1; then \
	  rm ${TRIPLES_FILE}; \
	  echo -e "Deleted triples file at ${TRIPLES_FILE}"; \
	else \
	  echo -e "No triples file found at ${TRIPLES_FILE}"; \
	fi

download-all: download-wikidata-mappings download-type-features download-predicate-variances

download-wikidata-mappings:
	@[ -d ${WIKIDATA_MAPPINGS_DIR} ] || mkdir ${WIKIDATA_MAPPINGS_DIR}
	wget https://ad-research.cs.uni-freiburg.de/data/natural-entity-types/wikidata_mappings.tar.gz
	tar -xvzf wikidata_mappings.tar.gz -C ${WIKIDATA_MAPPINGS_DIR}
	rm wikidata_mappings.tar.gz

download-type-features:
	@[ -d ${TYPE_FEATURES_DIR} ] || mkdir ${TYPE_FEATURES_DIR}
	wget https://ad-research.cs.uni-freiburg.de/data/natural-entity-types/computed_type_features.tar.gz
	tar -xvzf computed_type_features.tar.gz -C ${TYPE_FEATURES_DIR}
	rm computed_type_features.tar.gz

download-predicate-variances:
	@[ -d ${PREDICATE_VARIANCES_DIR} ] || mkdir ${PREDICATE_VARIANCES_DIR}
	wget https://ad-research.cs.uni-freiburg.de/data/natural-entity-types/predicate_variances.tar.gz
	tar -xvzf predicate_variances.tar.gz -C ${PREDICATE_VARIANCES_DIR}
	rm predicate_variances.tar.gz

generate-all: generate-wikidata-mappings compute-type-features compute-predicate-variances

generate-wikidata-mappings: get-qlever-mappings generate-databases

get-qlever-mappings:
	@echo
	@echo "[get-qlever-mappings] Get data for given queries from QLever."
	@echo
	@echo "DATA_QUERY_NAMES = $(DATA_QUERY_NAMES)"
	@[ -d ${WIKIDATA_MAPPINGS_DIR} ] || mkdir ${WIKIDATA_MAPPINGS_DIR}
	for QUERY_NAME in $(DATA_QUERY_NAMES); do echo; \
	  echo $${QUERY_NAME}; \
	  LOWER_QUERY_NAME=$$(echo $${QUERY_NAME} | tr '[:upper:]' '[:lower:]'); \
	  $(MAKE) -sB API=$${WIKIDATA_SPARQL_ENDPOINT} QUERY_VARIABLE=$${QUERY_NAME}_QUERY OUTFILE=$${WIKIDATA_MAPPINGS_DIR}$${LOWER_QUERY_NAME}.tsv query; done
	@echo

generate-databases:
	@echo
	@echo "[generate-databases] Build databases from large Wikidata mappings."
	@echo
	python3 scripts/create_databases.py ${WIKIDATA_MAPPINGS_DIR}qid_to_label.tsv
	python3 scripts/create_databases.py ${WIKIDATA_MAPPINGS_DIR}qid_to_description.tsv
	python3 scripts/create_databases.py ${WIKIDATA_MAPPINGS_DIR}qid_to_sitelinks.tsv
	python3 scripts/create_databases.py ${WIKIDATA_MAPPINGS_DIR}qid_to_p31.tsv -f multiple_values
	python3 scripts/create_databases.py ${WIKIDATA_MAPPINGS_DIR}qid_to_p279.tsv -f multiple_values

compute-type-features:
	@echo
	@echo "[compute-type-features] Compute feature mappings from wikidata mappings."
	@echo
	python3 scripts/get_type_frequencies_and_popularities.py

compute-predicate-variances:
	@echo
	@echo "[compute-predicate-variances] Compute predicate variances for various types"
	@echo
	python3 scripts/get_predicate_variance.py -qid all


# Get results for $(QUERY), convert to tsv and append to $(OUTFILE)
#
# Short descriptions of what the 3 sed lines do:
# 0) Replace wikidata entity URIs by the QID
# 1) Drop lines that don't start with Q
#    (e.g. Wikidata properties or lexemes or the first line of the file with column titles that start with "?")
# 2) Replace "<string>"@en for string literals by just <string>
# 3) Replace integer literals by just the integer
# 4) Remove the <> around wikipedia urls
query:
	@echo "API = ${API}"
	@echo "OUTFILE = ${OUTFILE}"
	@echo "$$PREFIXES $${${QUERY_VARIABLE}}"
	@curl -Gs ${API} -H "Accept: text/tab-separated-values"\
	    --data-urlencode "query=$$PREFIXES $${${QUERY_VARIABLE}} LIMIT 200000000" \
	    > ${DATA_DIR}tmp.tsv
	@cat ${DATA_DIR}tmp.tsv \
	    | sed -r 's|<http://www\.wikidata\.org/entity/([Q][0-9]+)>|\1|g' \
	    | sed -r '/^[^Q]/d' \
	    | sed -r 's|"([^\t"]*)"@en|\1|g' \
	    | sed -r 's|"([0-9][0-9]*)"\^\^<http://www\.w3\.org/2001/XMLSchema#int>|\1|g' \
	    | sed -r 's|<(http[s]*://[^\t ]*)>|\1|g' \
	    > ${OUTFILE}
	@echo "Number of lines in ${OUTFILE}:"
	@wc -l ${OUTFILE} | cut -f 1 -d " "
	@echo "First and last line:"
	@head -1 ${OUTFILE} && tail -1 ${OUTFILE}
	@rm -f ${DATA_DIR}tmp.tsv



define PREFIXES
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
endef

define QID_TO_LABEL_QUERY
SELECT DISTINCT ?item ?label WHERE {
  ?item @en@rdfs:label ?label
}
endef

define QID_TO_SITELINKS_QUERY
SELECT ?s ?o WHERE {
  ?sn schema:about ?s .
  ?sn wikibase:sitelinks ?o
  FILTER(?o > 0)
} ORDER BY DESC(?o)
endef

define QID_TO_DESCRIPTION_QUERY
SELECT DISTINCT ?item ?description WHERE {
  ?item @en@schema:description ?description .
  ?item_node schema:about ?item .
}
endef

define QID_TO_P31_QUERY
SELECT ?item ?type WHERE {
    ?item wdt:P31 ?type .
}
endef

define QID_TO_P279_QUERY
SELECT ?item ?type WHERE {
    ?item wdt:P279 ?type .
}
endef

export
