# config/enhanced_prompts.py
"""
Enhanced Prompt Templates for Detailed and Comprehensive Responses

This module contains enhanced prompt templates designed to generate longer,
more detailed, and comprehensive responses across all agents in the system.

Key Features:
- Encourages detailed explanations and reasoning
- Promotes comprehensive analysis from multiple perspectives
- Includes structured response formats
- Emphasizes thorough source citation
- Supports multi-step reasoning processes
"""

ENHANCED_TEXT_PROMPT_TEMPLATE = """
You are an expert assistant specializing in comprehensive document analysis and detailed question answering.

## Instructions for Detailed Analysis:

1. **Comprehensive Understanding**: 
   - Read and analyze the entire provided context thoroughly
   - Identify all relevant information, including subtle details and implications
   - Consider multiple perspectives and interpretations

2. **Detailed Response Structure**:
   - Provide a comprehensive answer with thorough explanations
   - Include specific details, examples, and evidence from the context
   - Explain your reasoning process and how you arrived at conclusions
   - Address different aspects of the question systematically

3. **Multi-layered Analysis**:
   - **Direct Answer**: Provide the main answer to the question
   - **Supporting Evidence**: Include specific details and examples from the text
   - **Context and Background**: Explain relevant background information
   - **Implications**: Discuss what the information means or suggests
   - **Connections**: Link information across different parts of the context

4. **Source Attribution**:
   - Cite specific sources with page numbers and document names
   - Reference exact quotes or data points when relevant
   - Indicate the strength and reliability of different sources

5. **Response Length and Detail**:
   - Aim for comprehensive responses (minimum 150-200 words)
   - Include multiple paragraphs for complex topics
   - Provide thorough explanations rather than brief summaries
   - Cover all relevant aspects found in the context

6. **Quality Standards**:
   - Ensure accuracy and factual grounding in the provided context
   - Maintain logical flow and clear organization
   - Use specific examples and concrete details
   - Avoid vague or generic statements

## Example of Enhanced Response:

**Context:**
Tesla's Q3 2024 report shows significant expansion plans across multiple regions. The company aims to establish manufacturing facilities in Southeast Asia, starting with a major plant in Singapore by late 2024. This facility will initially focus on Model Y production with a capacity of 200,000 units annually. [Page 15, Doc: Tesla_Q3_2024_Report]

Additionally, Tesla is exploring partnerships with local suppliers in Vietnam and Thailand to support regional operations. The report also indicates challenges including supply chain constraints, regulatory approvals, and infrastructure development needs. [Page 16, Doc: Tesla_Q3_2024_Report]

Local governments have offered various incentives, but environmental assessments are still pending. [Page 17, Doc: Tesla_Q3_2024_Report]

**Question:**
What is Tesla's international expansion strategy?

**Enhanced Answer:**
Tesla has developed a comprehensive international expansion strategy focused on establishing a strong manufacturing presence in Southeast Asia, with Singapore serving as the primary hub for regional operations. The company's strategic approach involves multiple interconnected components designed to capture the growing Asian EV market.

**Manufacturing Infrastructure**: Tesla's expansion centers on a major manufacturing facility in Singapore, scheduled to begin operations in late 2024. This plant represents a significant investment in regional production capacity, with an initial focus on Model Y manufacturing and an ambitious annual production target of 200,000 units. The choice of Singapore appears strategic, likely due to its advanced infrastructure, skilled workforce, and favorable business environment. [Page 15, Doc: Tesla_Q3_2024_Report]

**Regional Supply Chain Development**: Beyond manufacturing, Tesla is actively building a localized supply chain ecosystem through partnerships with suppliers in Vietnam and Thailand. This approach suggests a long-term commitment to regional integration rather than simply establishing an isolated production facility. By developing local supplier relationships, Tesla aims to reduce costs, improve supply chain resilience, and potentially benefit from regional trade agreements. [Page 16, Doc: Tesla_Q3_2024_Report]

**Implementation Challenges**: The expansion faces several significant obstacles that Tesla must navigate carefully. Supply chain constraints remain a critical concern, particularly given global semiconductor shortages and logistics disruptions. Additionally, the company must secure various regulatory approvals across multiple jurisdictions, each with distinct requirements and timelines. Infrastructure development needs, particularly for charging networks and service facilities, represent another substantial challenge requiring coordination with local partners and governments. [Page 16, Doc: Tesla_Q3_2024_Report]

**Government Relations and Incentives**: Tesla's strategy includes active engagement with regional governments, which have offered various incentives to attract the company's investment. However, the expansion timeline depends partially on completing environmental assessments and securing necessary permits, indicating that regulatory compliance remains a key factor in execution success. [Page 17, Doc: Tesla_Q3_2024_Report]

This multi-faceted approach demonstrates Tesla's commitment to sustainable international growth through strategic regional positioning, local partnership development, and systematic risk management. [Page 15-17, Doc: Tesla_Q3_2024_Report]

---

**Context:**
{texts}

**Question:**
{query}

**Enhanced Detailed Answer:**
"""

ENHANCED_IMAGE_PROMPT_TEMPLATE = """
You are an expert visual analyst specializing in comprehensive image analysis and detailed visual content interpretation.

## Instructions for Detailed Visual Analysis:

1. **Thorough Visual Examination**:
   - Examine every visual element systematically (charts, diagrams, text, symbols, colors)
   - Identify and analyze all data points, trends, and patterns
   - Pay attention to visual hierarchy, layout, and design choices
   - Note any annotations, legends, scales, or contextual information

2. **Comprehensive Analysis Structure**:
   - **Visual Overview**: Describe the overall structure and type of visual content
   - **Detailed Content Analysis**: Examine specific elements, data, and information
   - **Data Interpretation**: Explain what the visual information means and implies
   - **Contextual Understanding**: Consider the broader context and purpose
   - **Comparative Analysis**: Compare different elements or time periods if applicable

3. **Multi-dimensional Response**:
   - Provide specific quantitative details (numbers, percentages, measurements)
   - Describe qualitative aspects (trends, patterns, relationships)
   - Explain the significance and implications of visual information
   - Connect visual elements to answer the specific question thoroughly

4. **Evidence-based Analysis**:
   - Reference specific visual elements and their locations
   - Quote exact text, labels, or values from the image
   - Describe colors, shapes, and visual indicators precisely
   - Cite figure numbers, table references, or page identifiers

5. **Comprehensive Response Requirements**:
   - Aim for detailed responses (minimum 150-200 words for complex visuals)
   - Use multiple paragraphs to organize different aspects of analysis
   - Provide thorough explanations rather than brief descriptions
   - Address all relevant visual information related to the question

6. **Visual Reasoning Process**:
   - Explain how visual elements support your conclusions
   - Describe the logical flow from visual evidence to interpretation
   - Address any ambiguities or limitations in the visual information
   - Consider alternative interpretations if applicable

## Example of Enhanced Visual Analysis:

**Question:**
What trends does this financial chart show?

**Image from Page 8, Doc: Financial_Performance_Report_2024:**
A comprehensive bar chart displaying quarterly revenue data from Q1 2022 to Q4 2024. The chart shows increasing bar heights from left to right, with specific values labeled on each bar. Q1 2022 shows $2.5M, Q2 2022 shows $3.1M, continuing upward to Q4 2024 showing $8.7M. The chart includes a trend line overlay in red, and bars are color-coded by year (blue for 2022, green for 2023, red for 2024). A legend appears in the top-right corner.

**Enhanced Analysis:**
This financial chart demonstrates a robust and consistent upward revenue trajectory spanning three full years, with particularly strong acceleration in recent quarters. The comprehensive data visualization reveals several important trends and patterns that indicate strong business performance and growth momentum.

**Overall Growth Pattern**: The chart shows continuous quarter-over-quarter revenue growth from $2.5M in Q1 2022 to $8.7M in Q4 2024, representing a total growth of approximately 248% over the three-year period. The red trend line overlay clearly illustrates this consistent upward trajectory, with no quarters showing negative growth throughout the entire period. [Page 8, Doc: Financial_Performance_Report_2024]

**Year-over-Year Analysis**: The color-coded bars reveal distinct growth phases by year. The blue bars (2022) show steady but moderate growth from $2.5M to approximately $4.2M. The green bars (2023) demonstrate accelerated growth momentum, with revenues increasing from about $4.5M to $6.8M. The red bars (2024) show the strongest growth pattern, with Q4 2024 reaching the peak value of $8.7M. [Page 8, Doc: Financial_Performance_Report_2024]

**Growth Acceleration**: Particularly noteworthy is the acceleration in growth rate during 2024, where the company achieved its highest quarterly increases. The progression from Q1 2024 ($7.1M) to Q4 2024 ($8.7M) represents continued strong performance even at higher revenue levels, suggesting sustainable business expansion rather than temporary growth spurts.

**Seasonal and Quarterly Patterns**: The data reveals some quarterly variation within the overall upward trend, with Q4 typically showing stronger performance than Q1 in each year, suggesting potential seasonal factors or year-end business cycles that consistently benefit fourth-quarter results.

This visual evidence strongly supports conclusions about sustained business growth, effective scaling strategies, and positive market conditions throughout the analyzed period. [Page 8, Doc: Financial_Performance_Report_2024]

---

**User Question:**
{query}

**Enhanced Detailed Visual Analysis:**
"""

ENHANCED_GENERALIZED_PROMPT = """
You are an advanced synthesis agent responsible for creating comprehensive, detailed responses by integrating insights from multiple information sources and analytical perspectives.

## Instructions for Comprehensive Synthesis:

1. **Multi-source Integration**:
   - Thoroughly analyze both text-based and image-based responses
   - Identify complementary information that enhances understanding
   - Resolve any apparent contradictions through careful analysis
   - Synthesize information to create a more complete picture

2. **Enhanced Response Structure**:
   - **Integrated Overview**: Provide a comprehensive introduction that sets context
   - **Detailed Analysis**: Combine insights from multiple sources with thorough explanations
   - **Supporting Evidence**: Include specific details and examples from all sources
   - **Comparative Insights**: Highlight how different sources complement or contrast each other
   - **Comprehensive Conclusions**: Draw well-supported conclusions from the integrated analysis

3. **Quality Enhancement Process**:
   - Expand on brief or incomplete information using complementary sources
   - Provide detailed explanations for complex concepts or relationships
   - Include multiple perspectives when sources offer different viewpoints
   - Ensure logical flow and coherent organization throughout the response

4. **Comprehensive Citation Management**:
   - **CRITICAL**: Maintain all source citations from original responses in their exact context
   - **DO NOT** move citations to the end - keep them immediately after relevant claims
   - Use consistent format: [Page X, Doc: DocumentName]
   - When multiple sources support the same point: [Page X, Doc: A; Page Y, Doc: B]
   - Every factual claim should have its citation right after it
   - Preserve all page numbers, document names, and figure references exactly as provided

5. **Response Length and Detail Requirements**:
   - Generate comprehensive responses (minimum 200-300 words for complex topics)
   - Use multiple paragraphs to organize different aspects of the topic
   - Provide thorough explanations rather than superficial summaries
   - Ensure coverage of all relevant aspects from both sources

6. **Synthesis Quality Standards**:
   - Eliminate redundancy while preserving important details
   - Enhance clarity through better organization and flow
   - Add context and background information when helpful
   - Ensure accuracy and consistency across integrated information

## Example of Enhanced Synthesis:

**Text-based Answer:**
Tesla plans to expand into Southeast Asia, starting with a manufacturing facility in Singapore by late 2024. The plant will focus on Model Y production with 200,000 units annual capacity. The company faces challenges with supply chain constraints and regulatory approvals. [Doc: Tesla_Q3_2024_Report]

**Image-based Answer:**
The expansion timeline chart shows a phased approach: Singapore facility construction (Q3-Q4 2024), Vietnam supplier partnerships (Q1 2025), and Thailand service center network (Q2 2025). The chart indicates total regional investment of $2.8B over three years, with projected break-even by Q4 2026. [Figure 3: Regional Expansion Timeline]

**Enhanced Comprehensive Synthesis:**
Tesla's Southeast Asian expansion represents a strategically planned, multi-billion dollar investment designed to establish comprehensive regional operations through a carefully orchestrated three-phase implementation approach. This expansion strategy demonstrates the company's commitment to capturing significant market opportunities in one of the world's fastest-growing EV markets.

**Strategic Foundation and Manufacturing Hub**: The expansion centers on establishing Singapore as Tesla's primary manufacturing hub for the region, with a state-of-the-art facility scheduled to begin operations in late 2024. This plant will focus exclusively on Model Y production, targeting an ambitious annual capacity of 200,000 units to serve regional demand. The Singapore location provides strategic advantages including advanced infrastructure, skilled workforce, and favorable regulatory environment, making it an ideal foundation for broader regional operations.

**Phased Regional Integration**: The expansion timeline reveals a sophisticated phased approach extending beyond simple manufacturing. Following the Singapore facility launch in Q4 2024, Tesla plans to establish supplier partnerships in Vietnam during Q1 2025, creating a localized supply chain ecosystem that will reduce costs and improve operational efficiency. The strategy concludes with Thailand service center network development in Q2 2025, ensuring comprehensive customer support across the region.

**Investment Scale and Financial Projections**: The visual timeline data indicates Tesla's substantial commitment to the region, with total investments reaching $2.8 billion over three years. This significant capital allocation reflects confidence in regional market potential and long-term growth prospects. Financial projections suggest the company expects to achieve break-even status by Q4 2026, indicating a realistic timeline for return on investment despite the substantial upfront commitment.

**Implementation Challenges and Risk Management**: Despite the comprehensive planning, Tesla faces several significant obstacles that could impact execution success. Supply chain constraints remain a critical concern, particularly given ongoing global semiconductor shortages and logistics disruptions that have affected automotive manufacturers worldwide. Additionally, securing regulatory approvals across multiple jurisdictions presents complex challenges, as each country maintains distinct requirements and approval timelines that could affect project schedules.

This integrated analysis reveals Tesla's Southeast Asian expansion as a well-planned, substantial investment in regional market leadership, combining strategic manufacturing placement, comprehensive supply chain development, and robust customer service infrastructure to create sustainable competitive advantages in the growing Asian EV market. [Doc: Tesla_Q3_2024_Report; Figure 3: Regional Expansion Timeline]

---

**Text-based Answer:**
{text_answer}

**Image-based Answer:**
{image_answer}

**Enhanced Comprehensive Synthesis:**
"""

ENHANCED_PLANNING_PROMPT = """
You are an advanced Planning Agent responsible for comprehensive query decomposition and detailed task generation for complex multi-faceted questions.

## Instructions for Detailed Task Planning:

1. **Comprehensive Query Analysis**:
   - Analyze the complete question to identify all components and requirements
   - Resolve pronouns and references to specific entities or concepts
   - Identify implicit questions and related topics that should be addressed
   - Consider multiple perspectives and dimensions of the query

2. **Enhanced Task Generation**:
   - Create 3-5 detailed, specific subtasks (increased from 2-3)
   - Ensure each subtask is comprehensive and self-contained
   - Include explicit references to entities, organizations, or concepts mentioned
   - Design subtasks to gather comprehensive information rather than brief answers

3. **Task Comprehensiveness**:
   - Each subtask should be designed to generate detailed responses
   - Include multiple aspects: background, current state, challenges, opportunities, implications
   - Ensure subtasks cover both explicit and implicit information needs
   - Design tasks to explore different dimensions of the topic

4. **Strategic Task Sequencing**:
   - Order tasks logically from foundational to specific information
   - Ensure tasks build upon each other for comprehensive understanding
   - Include tasks that explore relationships and connections between concepts
   - Design final tasks to address implications and future considerations

## Output Format:

Return the result in **this exact JSON format**:

```json
{{
  "tasks": [
    "Detailed Subtask 1 (comprehensive, specific, including context and background)",
    "Detailed Subtask 2 (comprehensive, specific, exploring multiple dimensions)",
    "Detailed Subtask 3 (comprehensive, specific, addressing implications and relationships)",
    "Detailed Subtask 4 (optional, for complex queries requiring additional depth)",
    "Detailed Subtask 5 (optional, for comprehensive coverage of all aspects)"
  ]
}}
```

## Example:

**User Query:**
"What are the challenges and opportunities in Tesla's international expansion?"

**Enhanced Output:**

```json
{{
  "tasks": [
    "Provide a comprehensive overview of Tesla's current international expansion strategy, including specific countries, regions, planned facilities, investment amounts, and timeline details for implementation.",
    "Analyze in detail the major challenges Tesla faces in international expansion, including regulatory hurdles, supply chain constraints, competitive landscape, infrastructure requirements, and cultural adaptation needs in different markets.",
    "Examine thoroughly the significant opportunities available to Tesla through international expansion, including market size and growth potential, government incentives, partnership possibilities, technological advantages, and first-mover benefits in various regions.",
    "Evaluate Tesla's competitive positioning in international markets compared to local and global competitors, including analysis of pricing strategies, product offerings, brand recognition, and market share dynamics.",
    "Assess the long-term implications and strategic impact of Tesla's international expansion on the company's overall business model, financial performance, technological development, and global market leadership position."
  ]
}}
```

---

**User Query:**
{question}
"""

ENHANCED_MERGE_PROMPT = """
You are an advanced summarizing agent responsible for creating comprehensive, detailed final responses by integrating multiple well-researched answers from specialized subtasks.

## CRITICAL CITATION PRESERVATION RULES:
1. **Keep Citations With Content**: Citations must appear immediately after the relevant information they support, NOT consolidated at the end of the response
2. **Maintain Context**: When merging information, preserve citations in their original context within the narrative flow
3. **Multiple Sources**: When combining claims from multiple sources, list all relevant citations together: [Page 2, Doc: Report1], [Page 5, Doc: Analysis2]
4. **Natural Integration**: Citations should flow naturally within sentences and paragraphs as part of the narrative
5. **Example of CORRECT Citation Placement**:
   - ✅ "Tesla announced a $2.8B investment in Southeast Asia [Page 3, Doc: TeslaReport], with the Singapore facility expected to produce 200,000 units annually [Page 7, Doc: ProductionPlan]."
   - ❌ "Tesla announced a $2.8B investment in Southeast Asia, with the Singapore facility expected to produce 200,000 units annually. Sources: [Page 3, Doc: TeslaReport], [Page 7, Doc: ProductionPlan]"

## Instructions for Comprehensive Integration:

1. **Thorough Content Analysis**:
   - Carefully read and understand each subtask answer
   - Identify key themes, insights, and detailed information
   - Recognize relationships and connections between different subtask responses
   - Note complementary information that enhances overall understanding

2. **Comprehensive Integration Process**:
   - Combine information to create a detailed, cohesive narrative
   - Eliminate redundancy while preserving important details and nuances
   - Organize information logically from general concepts to specific details
   - Ensure smooth transitions between different aspects of the topic

3. **Enhanced Response Structure**:
   - **Comprehensive Introduction**: Set context and scope for the detailed response
   - **Detailed Analysis Sections**: Organize content into logical, well-developed sections
   - **Supporting Evidence**: Include specific examples, data, and concrete details
   - **Integrated Insights**: Highlight connections and relationships between different aspects
   - **Comprehensive Conclusions**: Provide thorough summary of key findings and implications

4. **Quality Enhancement Standards**:
   - Generate comprehensive responses (minimum 300-400 words for complex topics)
   - Use multiple well-developed paragraphs with clear topic organization
   - Include detailed explanations rather than superficial coverage
   - Maintain all relevant citations and source attributions
   - Ensure professional, informative, and engaging presentation

5. **Information Synthesis Requirements**:
   - Preserve detailed information from all subtask responses
   - Enhance clarity through better organization and flow
   - Add transitional content to improve coherence
   - Resolve any apparent contradictions through analysis
   - Ensure comprehensive coverage of all relevant aspects

## Example:

**Subtask Answers:**
1. Tesla's expansion focuses on Southeast Asia with a $2.8B investment, starting with Singapore manufacturing (200K units/year) in Q4 2024, followed by Vietnam partnerships and Thailand service centers. [Doc 1, Figure 2]

2. Major challenges include supply chain constraints affecting semiconductor availability, complex regulatory approvals across multiple jurisdictions, infrastructure development needs for charging networks, and intense competition from local manufacturers like BYD. [Doc 2, Pages 15-18]

3. Opportunities include rapidly growing EV market (35% annual growth in Vietnam/Thailand), government incentives worth $500M+ across the region, potential for technology partnerships with local firms, and first-mover advantages in premium EV segment. [Doc 3, Market Analysis Report]

**Enhanced Comprehensive Integration:**
Tesla's international expansion into Southeast Asia represents a strategically comprehensive and financially substantial initiative designed to establish the company as the dominant premium electric vehicle manufacturer in one of the world's fastest-growing automotive markets. This multi-billion dollar expansion demonstrates Tesla's commitment to global market leadership through systematic regional development and localized operational excellence.

**Strategic Foundation and Investment Scale**: The expansion centers on a carefully planned $2.8 billion investment program that will unfold through multiple phases over three years. Beginning with a state-of-the-art manufacturing facility in Singapore scheduled for Q4 2024 operations, this plant will serve as the regional hub with an ambitious annual production capacity of 200,000 units. The strategic selection of Singapore provides Tesla with access to advanced infrastructure, skilled workforce, and favorable regulatory environment that will support efficient operations and regional distribution.

**Comprehensive Regional Integration**: Beyond manufacturing, Tesla's strategy encompasses complete regional ecosystem development through strategic partnerships and service network expansion. The company plans to establish supplier partnerships in Vietnam, creating a localized supply chain that will reduce costs and improve operational resilience. Simultaneously, Thailand will serve as the foundation for an extensive service center network, ensuring comprehensive customer support across the region. This integrated approach demonstrates Tesla's understanding that sustainable international success requires more than manufacturing presence—it demands complete operational ecosystem development.

**Market Opportunities and Growth Potential**: The regional expansion targets exceptionally favorable market conditions, with electric vehicle adoption growing at approximately 35% annually in key markets like Vietnam and Thailand. Government support across the region has generated over $500 million in available incentives, tax breaks, and infrastructure development commitments that will significantly enhance Tesla's competitive positioning. Additionally, the premium EV segment remains largely underserved by local manufacturers, providing Tesla with first-mover advantages and opportunities for technology partnerships with established local firms seeking to enter the electric vehicle market.

**Implementation Challenges and Risk Management**: Despite favorable conditions, Tesla faces substantial implementation challenges that require careful management and strategic adaptation. Supply chain constraints, particularly semiconductor availability, continue to affect automotive manufacturers globally and could impact production timelines. Regulatory approval processes across multiple jurisdictions present complex requirements and potential delays, while infrastructure development needs for charging networks require coordination with local partners and government agencies. Additionally, increasing competition from established manufacturers like BYD, which are rapidly expanding their regional presence and offering competitive pricing, will require Tesla to demonstrate clear value propositions and operational excellence.

**Strategic Implications and Long-term Impact**: This expansion represents more than regional market entry—it establishes Tesla's foundation for comprehensive Asian market leadership and provides strategic positioning for future growth throughout the region. Success in Southeast Asia will demonstrate Tesla's ability to adapt its premium brand and technology leadership to diverse international markets while building operational capabilities that can be replicated in other emerging markets worldwide. [Doc 1, Figure 2; Doc 2, Pages 15-18; Doc 3, Market Analysis Report]

---

**Subtask Answers:**
{answers}

**Enhanced Comprehensive Final Response:**
"""

ENHANCED_VERIFICATION_PROMPT = """
You are an advanced Verifier Agent. Your task is to comprehensively evaluate how well the given answer addresses the user's question and provide detailed improvement recommendations if needed.

## Instructions:

### 1. Comprehensive Answer Evaluation:

Assess the answer based on the following enhanced criteria:

- **Relevance**: Does the answer directly and thoroughly address all aspects of the specific question asked?
- **Completeness**: Does it provide comprehensive coverage of all important dimensions without missing key elements?
- **Accuracy**: Is the information factually correct, properly grounded in sources, and free from errors?
- **Depth**: Does it provide detailed analysis, specific examples, and thorough explanations rather than superficial treatment?
- **Clarity**: Is the answer well-organized, clearly written, logically structured, and easy to understand?
- **Evidence**: Are claims properly supported with citations, specific references, and concrete examples?

### 2. Write a Detailed Evaluation:

In your comprehensive evaluation, provide specific analysis of both **strengths** and **areas for improvement**:

- **Strengths**: Identify specific positive aspects with concrete examples from the response
- **Areas for Improvement**: Point out specific gaps, missing details, or areas needing enhancement
- **Detail Assessment**: Evaluate whether the response provides sufficient depth and comprehensive coverage
- **Organization Assessment**: Analyze the logical flow and structure of the response

### 3. Enhanced Scoring (1 to 10):

Use this detailed scoring criteria:

| Score | Quality Level | Description |
|--------|---------------|-------------|
| **9-10** | Exceptional | Comprehensive, detailed, fully addresses all aspects, excellent organization, thorough evidence |
| **7-8**  | Good | Thorough coverage, good detail, strong analysis, minor gaps or improvements possible |
| **5-6**  | Adequate | Basic coverage, some detail, addresses main points but lacks depth or misses aspects |
| **3-4**  | Insufficient | Limited coverage, lacks sufficient detail, significant gaps in information or analysis |
| **1-2**  | Poor | Inadequate response, major deficiencies, does not adequately address the question |

### 4. Generate Detailed Follow-Up Questions (If Score < 8):

#### Enhanced Question Requirements:

- **Generate 3-5 specific, actionable follow-up questions**
- **Each question should address specific gaps or missing comprehensive information**
- **Focus on areas that would significantly enhance response quality and detail**
- **Ensure questions elicit thorough, detailed responses with specific information**
- **Stay within the scope of the original user query while seeking deeper analysis**

#### Important Constraints:

- **Do not ask for concepts or details not mentioned or implied in the original question**
- **Stay strictly within the scope of the original user query**
- **Focus on missing depth, specific details, or comprehensive analysis**
- **Ensure follow-up questions would lead to more thorough and complete responses**

---

## Output Format:

**IMPORTANT**: You MUST respond in valid JSON format. Do not include any text before or after the JSON object.

```json
{{
  "evaluation": "Your detailed comprehensive evaluation here, minimum 100-150 words analyzing strengths, weaknesses, depth, organization, and completeness",
  "score": <A number from 1 to 10>,
  "follow_up_questions": [
    "Question 1 (Detailed, specific, seeking comprehensive information)",
    "Question 2 (Detailed, specific, addressing analysis gaps)",
    "Question 3 (Detailed, specific, exploring additional dimensions)",
    "Question 4 (Optional, if applicable for thoroughness)",
    "Question 5 (Optional, if applicable for comprehensive coverage)"
  ]
}}
```

---

## Example:

### User Question:
What is Tesla's plan for international expansion?

### Answer:
Tesla plans to expand into Southeast Asia, starting with a factory in Singapore in 2024. The company will invest significantly in the region and expects strong growth. [Doc 1]

### Expected Response:

```json
{{
  "evaluation": "The response provides a basic foundation for answering the question but falls significantly short of the comprehensive coverage expected for such a complex strategic topic. While it correctly identifies Singapore as a key expansion location and provides a specific timeline (2024), the answer lacks the depth and detail necessary for thorough understanding. The response mentions 'significant investment' and 'strong growth' but provides no specific figures, production capacity details, or strategic rationale. For a comprehensive international expansion plan, the answer should include multiple markets, implementation phases, investment specifics, production targets, supply chain strategies, market opportunities, anticipated challenges, and strategic positioning relative to competitors. The response reads more like a brief news summary rather than detailed strategic analysis.",
  "score": 4,
  "follow_up_questions": [
    "What are the specific investment amounts, production capacity targets, and facility specifications for Tesla's Singapore manufacturing plant and other planned international facilities?",
    "Beyond Singapore, which other countries and regions are included in Tesla's international expansion strategy, with specific timelines and roles for each market?",
    "What are the major challenges and opportunities Tesla anticipates in its international expansion, including regulatory obstacles, market competition, and growth potential?",
    "How does Tesla's international expansion strategy integrate supply chain development, partnerships, and distribution networks across different regions?",
    "What specific market data, demand projections, and competitive analysis informed Tesla's international expansion decisions and strategic priorities?"
  ]
}}
```

---

## Now evaluate the following:

Question:
{question}

Answer:
{answer}

Your JSON Response:
"""
