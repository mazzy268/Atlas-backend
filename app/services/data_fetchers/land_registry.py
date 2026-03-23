"""
HM Land Registry — Price Paid Data
Free API, no key required.
Docs: https://landregistry.data.gov.uk/app/ppd
"""
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from app.core.logging import get_logger

log = get_logger(__name__)

HMLR_SPARQL_URL = "https://landregistry.data.gov.uk/landregistry/query"


SPARQL_BY_POSTCODE = """
PREFIX lrppi: <http://landregistry.data.gov.uk/def/ppi/>
PREFIX lrcommon: <http://landregistry.data.gov.uk/def/common/>

SELECT ?transactionId ?amount ?date ?propertyType ?newBuild ?tenure ?paon ?saon ?street ?town
WHERE {{
  ?trans lrppi:pricePaid ?amount ;
         lrppi:transactionDate ?date ;
         lrppi:propertyType ?propertyType ;
         lrppi:newBuild ?newBuild ;
         lrppi:estateType ?tenure ;
         lrppi:transactionId ?transactionId ;
         lrppi:propertyAddress ?addr .
  ?addr lrcommon:postcode "{postcode}" .
  OPTIONAL {{ ?addr lrcommon:paon ?paon }}
  OPTIONAL {{ ?addr lrcommon:saon ?saon }}
  OPTIONAL {{ ?addr lrcommon:street ?street }}
  OPTIONAL {{ ?addr lrcommon:town ?town }}
}}
ORDER BY DESC(?date)
LIMIT {limit}
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch(postcode: str, limit: int = 20) -> dict:
    """Fetch price-paid transactions for a postcode."""
    postcode_clean = postcode.strip().upper()
    query = SPARQL_BY_POSTCODE.format(postcode=postcode_clean, limit=limit)

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            HMLR_SPARQL_URL,
            params={"query": query, "output": "json"},
            headers={"Accept": "application/sparql-results+json"},
        )
        resp.raise_for_status()
        data = resp.json()

    bindings = data.get("results", {}).get("bindings", [])
    sales = []
    for b in bindings:
        sales.append({
            "transaction_id": _val(b, "transactionId"),
            "price_gbp": int(float(_val(b, "amount", "0"))),
            "date": _val(b, "date"),
            "property_type": _simplify_type(_val(b, "propertyType", "")),
            "new_build": _val(b, "newBuild", "false").lower() == "true",
            "tenure": _simplify_tenure(_val(b, "tenure", "")),
            "address_paon": _val(b, "paon"),
            "address_saon": _val(b, "saon"),
            "street": _val(b, "street"),
            "town": _val(b, "town"),
        })

    log.info("land_registry_fetched", postcode=postcode_clean, count=len(sales))
    return {"postcode": postcode_clean, "sales": sales, "total": len(sales)}


def _val(binding: dict, key: str, default: str = "") -> str:
    return binding.get(key, {}).get("value", default)


def _simplify_type(uri: str) -> str:
    mapping = {"detached": "Detached", "semi-detached": "Semi-detached",
               "terraced": "Terraced", "flat-maisonette": "Flat"}
    for k, v in mapping.items():
        if k in uri.lower():
            return v
    return uri.split("/")[-1] if uri else "Unknown"


def _simplify_tenure(uri: str) -> str:
    if "freehold" in uri.lower():
        return "Freehold"
    if "leasehold" in uri.lower():
        return "Leasehold"
    return uri.split("/")[-1] if uri else "Unknown"
