import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable


DEFAULT_CORPUS_PATH = Path("data") / "career_intelligence_corpus.json"


def clean_env(name, default=None):
    value = os.getenv(name, default)
    if value is None:
        return None
    return value.strip().strip('"').strip("'")


def build_uri(uri, trust_self_signed=False):
    if not uri:
        return uri
    uri = uri.strip()
    if trust_self_signed:
        return uri.replace("neo4j+s://", "neo4j+ssc://").replace("bolt+s://", "bolt+ssc://")
    return uri


class CareerGraphSeeder:
    def __init__(self, uri, user, password, database=None, trust_self_signed=False):
        self.database = database
        self.driver = self._connect(uri, user, password, trust_self_signed)

    @staticmethod
    def _protocol(uri):
        return uri.split("://", 1)[0] if uri and "://" in uri else "unknown"

    def _connect(self, uri, user, password, trust_self_signed):
        primary_uri = build_uri(uri, trust_self_signed=trust_self_signed)
        print(f"Connecting to Neo4j via {self._protocol(primary_uri)}://...")

        driver = GraphDatabase.driver(primary_uri, auth=(user, password))
        try:
            driver.verify_connectivity()
            return driver
        except ServiceUnavailable:
            driver.close()

            fallback_uri = build_uri(uri, trust_self_signed=True)
            if trust_self_signed or fallback_uri == primary_uri:
                raise

            print("Strict TLS routing failed; retrying with Neo4j self-signed certificate trust...")
            print(f"Connecting to Neo4j via {self._protocol(fallback_uri)}://...")
            fallback_driver = GraphDatabase.driver(fallback_uri, auth=(user, password))
            fallback_driver.verify_connectivity()
            return fallback_driver

    def close(self):
        self.driver.close()

    def session(self):
        if self.database:
            return self.driver.session(database=self.database)
        return self.driver.session()

    def seed_graph_from_corpus(self, corpus_path=DEFAULT_CORPUS_PATH, reset=True):
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Cannot find dataset file at: {corpus_path}")

        with corpus_path.open("r", encoding="utf-8") as file:
            corpus_data = json.load(file)

        print(f"Found {len(corpus_data)} Kaggle entities to process.")

        with self.session() as session:
            session.execute_write(self._seed_transaction, corpus_data, reset)

    @staticmethod
    def _seed_transaction(tx, corpus_data, reset):
        if reset:
            tx.run("MATCH (n) DETACH DELETE n").consume()
            print("Cleared existing graph records.")

        node_query = """
        MERGE (e:Entity {id: $id})
        SET e.title = $title,
            e.type = $type,
            e.description = $description,
            e.authority = "Kaggle Targeted Set"
        """

        for index, item in enumerate(corpus_data):
            tx.run(
                node_query,
                id=item.get("entity_id", f"kaggle_node_{1000 + index}"),
                title=item.get("title", f"Profile {index}"),
                type=item.get("type", "Role"),
                description=item.get("description", ""),
            )

        print("Phase 1: Nodes seeded.")

        link_query = """
        MATCH (a:Entity), (b:Entity)
        WHERE a <> b AND (
            toLower(a.description) CONTAINS toLower(b.title) OR
            toLower(b.description) CONTAINS toLower(a.title)
        )
        MERGE (a)-[r:REQUIRED_FOR]->(b)
        RETURN count(r) AS rel_count
        """
        rel_count = tx.run(link_query).single()["rel_count"]
        print(f"Phase 2: Formed {rel_count} graph relationships.")


def parse_args():
    parser = argparse.ArgumentParser(description="Seed Neo4j with the career intelligence corpus.")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS_PATH), help="Path to the corpus JSON file.")
    parser.add_argument("--no-reset", action="store_true", help="Do not delete existing graph data first.")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    uri = clean_env("NEO4J_URI")
    user = clean_env("NEO4J_USERNAME", "neo4j")
    password = clean_env("NEO4J_PASSWORD")
    database = clean_env("NEO4J_DATABASE")
    trust_self_signed = clean_env("NEO4J_TRUST_SELF_SIGNED", "false").lower() == "true"

    missing = [name for name, value in {
        "NEO4J_URI": uri,
        "NEO4J_USERNAME": user,
        "NEO4J_PASSWORD": password,
    }.items() if not value]

    if missing:
        print(f"Missing required environment values: {', '.join(missing)}")
        print("Update your .env file, then run this script again.")
        return 1

    database_label = database or "default"
    print(f"Using Neo4j username '{user}' and database '{database_label}'.")

    seeder = None
    try:
        seeder = CareerGraphSeeder(
            uri=uri,
            user=user,
            password=password,
            database=database,
            trust_self_signed=trust_self_signed,
        )
        seeder.seed_graph_from_corpus(args.corpus, reset=not args.no_reset)
        print("Cloud database syncing process complete.")
        return 0
    except AuthError:
        print("Neo4j rejected the username or password.")
        print("Check NEO4J_USERNAME and NEO4J_PASSWORD in .env, or reset the Aura password and paste the new value there.")
        return 1
    except ServiceUnavailable as exc:
        print(f"Could not reach Neo4j: {exc}")
        print("Check NEO4J_URI and make sure the Aura instance is running.")
        return 1
    except (FileNotFoundError, json.JSONDecodeError, Neo4jError) as exc:
        print(f"Seeding failed: {exc}")
        return 1
    finally:
        if seeder:
            seeder.close()


if __name__ == "__main__":
    raise SystemExit(main())