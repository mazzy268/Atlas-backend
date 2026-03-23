"""
All AI prompts used by the Atlas Property Intelligence report builder.
Each prompt is a function that accepts structured data and returns a prompt string.
Keeping prompts in one file makes them easy to iterate on independently.
"""


def investment_score_prompt(data: dict) -> str:
    return f"""You are a senior UK property investment analyst. Analyse the following data for a UK property and produce an Investment Score from 0 to 100.

PROPERTY DATA:
Address: {data.get('address')}
Recent sales (same postcode): {data.get('sales')}
EPC rating: {data.get('epc')}
Crime data: {data.get('crime')}
Flood risk: {data.get('flood')}
Schools nearby: {data.get('schools')}
Transport score: {data.get('transport')}
Planning applications nearby: {data.get('planning')}
Demographics: {data.get('demographics')}

Scoring criteria (weight each):
- Capital growth potential (25%)
- Rental yield potential (20%)
- Location desirability (20%)
- Risk profile: crime, flood, planning (20%)
- Infrastructure & amenities (15%)

Return a JSON object with exactly this structure:
{{
  "score": <integer 0-100>,
  "grade": <"A" if 80-100, "B" if 60-79, "C" if 40-59, "D" if 20-39, "F" if 0-19>,
  "reasoning": "<2-3 sentence explanation of score>",
  "key_positives": ["<positive 1>", "<positive 2>", "<positive 3>"],
  "key_risks": ["<risk 1>", "<risk 2>", "<risk 3>"]
}}

Return ONLY the JSON. No markdown, no explanation."""


def strategy_detector_prompt(data: dict) -> str:
    return f"""You are a UK property investment strategist. Based on the data below, identify the best investment strategies for this property.

PROPERTY DATA:
Address: {data.get('address')}
Recent sold prices nearby: {data.get('sales')}
EPC data (rooms, floor area): {data.get('epc')}
Crime score: {data.get('crime_summary')}
Transport links: {data.get('transport_summary')}
Demographics: {data.get('demographics')}
Planning history: {data.get('planning')}
Flood risk: {data.get('flood_risk')}

Available strategies to consider:
- BTL (Buy-to-Let): standard residential rental
- HMO (House in Multiple Occupation): room-by-room rental, higher yield
- SA (Serviced Accommodation / Airbnb): short-term lets
- Flip: buy, renovate, sell for profit
- BRRR (Buy, Refurb, Refinance, Rent): equity extraction strategy
- Commercial Conversion: convert to offices, retail
- Mixed Use: retain residential + add commercial ground floor

Return a JSON object with exactly this structure:
{{
  "primary_strategy": "<single best strategy name>",
  "recommended_strategies": ["<strategy 1>", "<strategy 2>", "<strategy 3>"],
  "reasoning": "<2-3 sentence explanation>",
  "estimated_yields": {{
    "BTL": "<e.g. 5.2%>",
    "HMO": "<e.g. 8.1%>",
    "SA": "<e.g. 11% (seasonal)>"
  }},
  "avoid_because": "<any strategies to avoid and why>"
}}

Return ONLY the JSON."""


def renovation_predictor_prompt(data: dict) -> str:
    return f"""You are a UK quantity surveyor and property developer. Estimate renovation costs and value uplift for this property.

PROPERTY DATA:
Address: {data.get('address')}
EPC details (construction, walls, roof, heating, windows): {data.get('epc')}
Property type from sales data: {data.get('property_type')}
Recent comparable sold prices: {data.get('comparables')}
Floor area (sqm): {data.get('floor_area')}
EPC current rating: {data.get('epc_rating')}
EPC potential rating: {data.get('epc_potential')}

Use BCIS (Building Cost Information Service) UK regional rates. Assume South East / London unless postcode suggests otherwise.

Estimate for a typical refurbishment:
- Light refurb: cosmetic (kitchen, bathrooms, decoration)
- Medium refurb: full refurb + EPC improvements
- Heavy refurb: structural work, extension potential

Return a JSON object with exactly this structure:
{{
  "current_estimated_value_gbp": <integer>,
  "light_refurb": {{
    "estimated_cost_gbp": <integer>,
    "after_repair_value_gbp": <integer>,
    "roi_pct": <float>,
    "payback_period_months": <integer>,
    "recommended_works": ["<work 1>", "<work 2>"]
  }},
  "medium_refurb": {{
    "estimated_cost_gbp": <integer>,
    "after_repair_value_gbp": <integer>,
    "roi_pct": <float>,
    "payback_period_months": <integer>,
    "recommended_works": ["<work 1>", "<work 2>"]
  }},
  "heavy_refurb": {{
    "estimated_cost_gbp": <integer>,
    "after_repair_value_gbp": <integer>,
    "roi_pct": <float>,
    "payback_period_months": <integer>,
    "recommended_works": ["<work 1>", "<work 2>"]
  }},
  "epc_upgrade_cost_gbp": <integer>,
  "epc_upgrade_notes": "<what work would get to EPC C or above>"
}}

Return ONLY the JSON."""


def floorplan_analysis_prompt(data: dict) -> str:
    return f"""You are a UK property architect and HMO licensing expert. Based on available data, assess this property's layout potential.

PROPERTY DATA:
Address: {data.get('address')}
EPC data (habitable rooms, floor area, property type, built form): {data.get('epc')}
Property type: {data.get('property_type')}
Number of habitable rooms from EPC: {data.get('habitable_rooms')}
Floor area sqm: {data.get('floor_area')}
Construction year (approx): {data.get('construction_year')}

Note: No actual floor plan image is available — base assessment on typical UK property layouts for this type/era/size.

Return a JSON object with exactly this structure:
{{
  "estimated_bedrooms": <integer>,
  "estimated_bathrooms": <integer>,
  "extension_potential": "<Rear extension likely / Side extension possible / Limited — terrace / None — flat>",
  "loft_conversion_viable": <true/false>,
  "loft_conversion_notes": "<brief note>",
  "hmo_room_potential": <integer — number of lettable rooms if converted to HMO>,
  "hmo_feasibility": "<High / Medium / Low>",
  "hmo_notes": "<key considerations — Article 4, parking, room sizes>",
  "layout_notes": "<2-3 sentences on typical layout for this property type and era>"
}}

Return ONLY the JSON."""


def neighbourhood_intelligence_prompt(data: dict) -> str:
    return f"""You are a UK property location analyst. Provide a comprehensive neighbourhood assessment.

DATA:
Address: {data.get('address')}
Crime statistics: {data.get('crime')}
Schools nearby: {data.get('schools')}
Transport data: {data.get('transport')}
ONS demographics: {data.get('demographics')}
Flood risk: {data.get('flood')}

Return a JSON object with exactly this structure:
{{
  "crime_score": <integer 0-10, 10 = safest>,
  "crime_summary": "<1-2 sentence plain English summary of crime levels>",
  "school_rating": "<Outstanding / Good / Mixed / Poor>",
  "best_nearby_school": "<name and Ofsted rating>",
  "school_notes": "<1 sentence>",
  "transport_score": <integer 0-10>,
  "transport_summary": "<1-2 sentence summary of transport links>",
  "income_estimate": "<e.g. 'Above average — median ~£42k'>",
  "deprivation_decile": <integer 1-10 or null, 1=most deprived>,
  "overall_desirability": "<Prime / Desirable / Average / Below average / Regeneration area>",
  "area_trajectory": "<Gentrifying / Stable / Declining / Regeneration zone>",
  "investor_appeal": "<High / Medium / Low — 1 sentence reason>"
}}

Return ONLY the JSON."""


def rental_demand_prompt(data: dict) -> str:
    return f"""You are a UK lettings market analyst. Score the rental demand for this property location.

DATA:
Address: {data.get('address')}
Demographics (working age population, employment rate): {data.get('demographics')}
Transport links: {data.get('transport_summary')}
Universities / large employers nearby: infer from address and demographics
Crime level: {data.get('crime_summary')}
Recent rental listings from area (if any): {data.get('recent_sales')}

Return a JSON object with exactly this structure:
{{
  "rental_demand_score": <integer 0-100>,
  "demand_level": "<Very High / High / Medium / Low / Very Low>",
  "key_tenant_profiles": ["<profile 1>", "<profile 2>"],
  "average_void_period_weeks": <integer>,
  "reasoning": "<2-3 sentences explaining the score>"
}}

Return ONLY the JSON."""


def planning_scanner_prompt(data: dict) -> str:
    return f"""You are a UK planning consultant. Analyse nearby planning activity and assess risks and opportunities.

DATA:
Address: {data.get('address')}
Nearby planning applications: {data.get('planning_applications')}
Property type: {data.get('property_type')}
Demographics / area trajectory: {data.get('demographics')}

Return a JSON object with exactly this structure:
{{
  "risk_level": "<Low / Medium / High>",
  "risk_summary": "<1-2 sentences on any planning risks — e.g. nearby high-rise, industrial>",
  "development_opportunity": "<what planning permissions are likely grantable for this property>",
  "article_4_risk": <true/false — is this likely an Article 4 direction area for HMOs?>,
  "permitted_development_likely": <true/false>,
  "notes": "<any other relevant planning intelligence>"
}}

Return ONLY the JSON."""


def deal_finder_prompt(data: dict) -> str:
    return f"""You are a UK property deal analyst. Assess whether this property represents a below-market opportunity.

DATA:
Address: {data.get('address')}
Recent comparable sold prices (same postcode/street): {data.get('comparables')}
EPC data (floor area, property type): {data.get('epc')}
Flood risk and other risk factors: {data.get('risk_factors')}
Any asking price provided: {data.get('asking_price', 'Not provided')}

Calculate price per sqm from comparables. Compare to this property.

Return a JSON object with exactly this structure:
{{
  "estimated_market_value_gbp": <integer>,
  "price_per_sqm_area_avg": <integer>,
  "is_below_market": <true/false>,
  "discount_pct": <float or null>,
  "deal_type": "<Below market / Fair value / Overpriced / Cannot determine>",
  "deal_score": <integer 0-10, 10 = exceptional deal>,
  "recommendation": "<Buy now / Worth investigating / Fair price — negotiate / Overpriced — avoid>",
  "notes": "<1-2 sentence justification>"
}}

Return ONLY the JSON."""


def price_growth_prompt(data: dict) -> str:
    return f"""You are a UK residential property economist. Forecast price growth for this property.

DATA:
Address: {data.get('address')}
Historical sold prices (this postcode, chronological): {data.get('sales_history')}
Demographics and area trajectory: {data.get('demographics')}
Transport improvements planned: {data.get('transport')}
Planning activity: {data.get('planning')}
Flood / environmental risks: {data.get('flood')}
Most recent sale price: {data.get('latest_price')}

UK long-run average house price growth: ~3.5% per annum nominal.
Adjust up/down based on: location quality, transport investment, regeneration, supply constraints, demand drivers.

Return a JSON object with exactly this structure:
{{
  "current_estimate_gbp": <integer>,
  "one_year_forecast_gbp": <integer>,
  "three_year_forecast_gbp": <integer>,
  "five_year_forecast_gbp": <integer>,
  "annual_growth_rate_pct": <float>,
  "confidence": "<Low / Medium / High>",
  "drivers": ["<driver 1>", "<driver 2>", "<driver 3>"],
  "risks_to_forecast": ["<downside risk 1>", "<downside risk 2>"],
  "methodology_note": "<1 sentence on approach used>"
}}

Return ONLY the JSON."""


def rental_yield_simulator_prompt(data: dict) -> str:
    return f"""You are a UK buy-to-let mortgage and cashflow analyst. Model the rental yield and cashflow for this property.

DATA:
Address: {data.get('address')}
Estimated property value: {data.get('estimated_value')}
EPC data (property size, type): {data.get('epc')}
Local rental demand score: {data.get('rental_demand_score')}
Demographics: {data.get('demographics')}

Assumptions to use (unless data suggests otherwise):
- Deposit: 25% (standard BTL)
- Mortgage rate: 5.5% (current UK BTL rate, 2024)
- Mortgage term: 25 years interest-only
- Letting agent fee: 10% of rent
- Maintenance allowance: 1% of property value per annum
- Void periods: 4 weeks per year
- Insurance: £500/year
- Landlord licence (if HMO applicable): £800/year

Return a JSON object with exactly this structure:
{{
  "estimated_monthly_rent_gbp": <integer>,
  "gross_yield_pct": <float>,
  "net_yield_pct": <float>,
  "monthly_mortgage_estimate_gbp": <integer>,
  "monthly_cashflow_gbp": <integer — positive means profit>,
  "annual_profit_gbp": <integer>,
  "annual_gross_rent_gbp": <integer>,
  "total_annual_costs_gbp": <integer>,
  "assumptions": {{
    "deposit_pct": 25,
    "mortgage_rate_pct": 5.5,
    "mortgage_type": "Interest-only",
    "letting_agent_fee_pct": 10,
    "void_weeks_per_year": 4,
    "maintenance_pct": 1.0
  }},
  "hmo_scenario": {{
    "estimated_monthly_rent_gbp": <integer — if converted to HMO>,
    "gross_yield_pct": <float>,
    "net_yield_pct": <float>,
    "monthly_cashflow_gbp": <integer>
  }},
  "notes": "<any important caveats>"
}}

Return ONLY the JSON."""


def ai_summary_prompt(data: dict) -> str:
    return f"""You are Atlas, an AI property investment assistant. Write a concise, professional investment summary for this UK property report.

FULL REPORT DATA:
Address: {data.get('address')}
Investment Score: {data.get('investment_score')}/100 (Grade {data.get('investment_grade')})
Primary Strategy: {data.get('primary_strategy')}
Estimated Value: £{data.get('estimated_value'):,}
Gross Yield: {data.get('gross_yield')}%
5-Year Price Forecast: +{data.get('five_year_growth')}%
Neighbourhood: {data.get('neighbourhood_summary')}
Key Risks: {data.get('key_risks')}
Key Positives: {data.get('key_positives')}

Write a 3-paragraph executive summary:
1. Paragraph 1: Property overview and investment score verdict
2. Paragraph 2: Best strategy and financial metrics (yield, cashflow, growth)
3. Paragraph 3: Key risks and due diligence recommendations

Tone: Professional, direct, data-driven. No hype. No disclaimers. Max 250 words total."""


def ai_assistant_prompt(property_context: dict, user_question: str) -> str:
    return f"""You are Atlas, an expert UK property investment AI assistant. Answer the user's question using the property data provided.

PROPERTY CONTEXT:
{property_context}

USER QUESTION: {user_question}

Rules:
- Be specific and data-driven — reference actual figures from the context
- If the answer requires data not in the context, say so clearly
- Keep answers concise (under 150 words unless the question demands more)
- Do not give regulated financial advice — say "speak to an independent financial adviser" where appropriate
- Use £ for currency, % for percentages
- Always answer in the context of UK property investment"""
