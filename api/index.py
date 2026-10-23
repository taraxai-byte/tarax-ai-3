"""
TARAX AI Hub v2: factory <-> buyer marketplace engine.
Env vars: ADMIN_KEY (required), ANTHROPIC_API_KEY (optional)
NOTE: in-memory store; swap for Postgres/Supabase before real use.
Escrow/payment is a REFERENCE to a licensed provider, never held by this code.
"""
import os, hmac, uuid
from datetime import date
from decimal import Decimal
from fastapi import FastAPI, APIRouter, HTTPException, Header
from pydantic import BaseModel

FEE_RATE = Decimal("0.0075")
MODEL = "claude-sonnet-5-5"
SANCTIONED = {"IR", "KP", "SY", "CU"}  # illustrative; use a real screening API
STATES = ["matched", "compliance_cleared", "escrow_funded", "shipped",
          "delivered", "buyer_confirmed", "settled"]


class Store:
    def __init__(self):
        self.companies, self.forecasts, self.rfqs, self.deals = {}, {}, {}, {}

    @staticmethod
    def new_id(prefix): return f"{prefix}-{uuid.uuid4().hex[:8]}"


db = Store()


class CompanyIn(BaseModel):
    name: str
    country: str
    role: str
    email: str


class ForecastIn(BaseModel):
    company_id: str
    product: str
    quantity: float
    unit: str
    ready_date: str
    min_price_usd: float


class RFQIn(BaseModel):
    company_id: str
    product: str
    quantity: float
    unit: str
    needed_by: str
    max_price_usd: float


class DealIn(BaseModel):
    forecast_id: str
    rfq_id: str


class AdvanceIn(BaseModel):
    to: str
    actor: str
    ref: str | None = None
    approved_by: str | None = None


app = FastAPI(title="TARAX AI Hub v2")
router = APIRouter()


def admin(key):
    expected = os.environ.get("ADMIN_KEY", "")
    if not expected or not hmac.compare_digest(key or "", expected):
        raise HTTPException(401, "Admin key required")


def get(coll, id_, label):
    if id_ not in coll:
        raise HTTPException(404, f"{label} not found")
    return coll[id_]


def norm(s): return " ".join(s.lower().split())


def d(s):
    try: return date.fromisoformat(s)
    except ValueError: raise HTTPException(422, f"Bad date '{s}', use YYYY-MM-DD")


def score(f: dict, r: dict) -> int:
    if norm(f["product"]) != norm(r["product"]) or f["unit"] != r["unit"]:
        return 0
    if f["min_price_usd"] > r["max_price_usd"]:
        return 0
    slack = (d(r["needed_by"]) - d(f["ready_date"])).days
    if slack < 0:
        return 0
    coverage = min(f["quantity"], r["quantity"]) / r["quantity"]
    headroom = min((r["max_price_usd"] - f["min_price_usd"]) / r["max_price_usd"], 1)
    timing = 1.0 if slack <= 90 else max(0.0, 1 - (slack - 90) / 180)
    return round(100 * (0.6 * coverage + 0.25 * headroom + 0.15 * timing))


async def explain(f, r, s) -> str:
    plain = f"Fit {s}/100: product matches, ready {f['ready_date']} before need-by {r['needed_by']}."
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return plain
    try:
        from anthropic import AsyncAnthropic
        m = await AsyncAnthropic().messages.create(
            model=MODEL, max_tokens=150,
            messages=[{"role": "user", "content":
                       f"One sentence, no hype, explain this match. Forecast={f} RFQ={r} score={s}"}])
        return m.content[0].text.strip()
    except Exception:
        return plain


def find_matches(forecast=None, rfq=None):
    out = []
    for f in ([forecast] if forecast else db.forecasts.values()):
        for r in ([rfq] if rfq else db.rfqs.values()):
            fc, bc = db.companies[f["company_id"]], db.companies[r["company_id"]]
            if not (fc["verified"] and bc["verified"]) or fc["country"] == bc["country"]:
                continue
            s = score(f, r)
            if s >= 50:
                out.append({"forecast_id": f["id"], "rfq_id": r["id"], "score": s})
    return sorted(out, key=lambda m: -m["score"])


def post_listing(coll, prefix, body: BaseModel, role):
    c = get(db.companies, body.company_id, "Company")
    if c["role"] != role:
        raise HTTPException(403, f"Only a {role} can post this")
    if not c["verified"]:
        raise HTTPException(403, "Company not verified yet")
    item = body.model_dump() | {"id": db.new_id(prefix)}
    coll[item["id"]] = item
    return item


@router.post("/companies")
async def register(c: CompanyIn):
    if c.role not in ("factory", "buyer"):
        raise HTTPException(422, "role must be factory or buyer")
    item = c.model_dump() | {"id": db.new_id("CO"), "verified": False}
    db.companies[item["id"]] = item
    return item


@router.post("/companies/{cid}/verify")
async def verify(cid: str, x_admin_key: str | None = Header(None)):
    admin(x_admin_key)
    get(db.companies, cid, "Company")["verified"] = True
    return db.companies[cid]


@router.post("/forecasts")
async def add_forecast(body: ForecastIn):
    d(body.ready_date)
    f = post_listing(db.forecasts, "FC", body, "factory")
    return {"forecast": f, "matches": find_matches(forecast=f)}


@router.post("/rfqs")
async def add_rfq(body: RFQIn):
    d(body.needed_by)
    r = post_listing(db.rfqs, "RFQ", body, "buyer")
    return {"rfq": r, "matches": find_matches(rfq=r)}


@router.get("/matches")
async def matches():
    return find_matches()


@router.post("/deals")
async def open_deal(body: DealIn):
    f = get(db.forecasts, body.forecast_id, "Forecast")
    r = get(db.rfqs, body.rfq_id, "RFQ")
    s = score(f, r)
    if s < 50:
        raise HTTPException(409, "Not a valid match")
    qty = min(f["quantity"], r["quantity"])
    price = (Decimal(str(f["min_price_usd"])) + Decimal(str(r["max_price_usd"]))) / 2
    deal = {"id": db.new_id("DEAL"), "forecast_id": f["id"], "rfq_id": r["id"],
            "factory": f["company_id"], "buyer": r["company_id"],
            "quantity": qty, "unit_price_usd": str(price.quantize(Decimal("0.0001"))),
            "value_usd": str((price * Decimal(str(qty))).quantize(Decimal("0.01"))),
            "status": "matched", "why": await explain(f, r, s), "log": []}
    db.deals[deal["id"]] = deal
    return deal


@router.post("/deals/{deal_id}/advance")
async def advance(deal_id: str, body: AdvanceIn, x_admin_key: str | None = Header(None)):
    deal = get(db.deals, deal_id, "Deal")
    cur = deal["status"]
    if body.to == "cancelled":
        if cur == "cancelled" or STATES.index(cur) >= STATES.index("shipped"):
            raise HTTPException(409, f"Cannot cancel from {cur}")
        deal["status"] = "cancelled"
    else:
        if cur not in STATES or body.to not in STATES or \
                STATES.index(body.to) != STATES.index(cur) + 1:
            raise HTTPException(409, f"Invalid move {cur} -> {body.to}")
        if body.to == "compliance_cleared":
            f, b = db.companies[deal["factory"]], db.companies[deal["buyer"]]
            if f["country"] in SANCTIONED or b["country"] in SANCTIONED:
                raise HTTPException(409, "Blocked: sanctioned jurisdiction")
        if body.to in ("escrow_funded", "shipped") and not body.ref:
            raise HTTPException(422, "ref required (escrow/shipment reference)")
        if body.to == "settled":
            admin(x_admin_key)
            if not body.approved_by:
                raise HTTPException(422, "approved_by required")
            fee = (Decimal(deal["value_usd"]) * FEE_RATE).quantize(Decimal("0.01"))
            deal |= {"tarax_fee_usd": str(fee),
                     "net_to_factory_usd": str(Decimal(deal["value_usd"]) - fee),
                     "approved_by": body.approved_by}
        deal["status"] = body.to
    deal["log"].append({"from": cur, "to": deal["status"], "actor": body.actor, "ref": body.ref})
    return deal


@router.get("/deals/{deal_id}")
async def get_deal(deal_id: str):
    return get(db.deals, deal_id, "Deal")


@router.get("/health")
async def health():
    return {"ok": True, "admin_key_set": bool(os.environ.get("ADMIN_KEY"))}


app.include_router(router)
app.include_router(router, prefix="/api")
