import google.generativeai as genai
import re
import os

# Load .env if present so GEMINI_API_KEY can be set there for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv not available; expect environment variables to be set externally
    pass

def process_financial_statement(pdf_path):
    """
    Process FAB Financial Statement PDF using Gemini
    Format: FAB-FS-{Quarter_number}-{year}-English.pdf
    """
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Please set it in the environment or in a .env file")
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    document = genai.upload_file(pdf_path)
    
    prompt = """
    You are processing a First Abu Dhabi Bank (FAB) Financial Statement PDF.
    
    EXTRACTION REQUIREMENTS:
    
    1. METADATA EXTRACTION:
       - Extract quarter (Q1, Q2, Q3, Q4) and year from filename/content
       - Document type: "financial_statement"
       - Report period end date
    
    2. SECTION PROCESSING:
       - For each major section, start with #Section [SECTION_NAME]
       - Include page number for each section: #Page [PAGE_NUMBER]
       - pay attention to the financial metrics corresponding to 2 column values such as totals, assets/liabilities, profit/loss, Equity etc.
       - pay attention to the alignment of numbers in tables espaecially when the headers are years
       - Preserve ALL financial tables with proper structure
       - Maintain number alignment and formatting
       - Extract both AED millions and percentages accurately
    
    3. COMMON SECTIONS TO EXPECT:
      Focus very well on parsing these sections accurately with proper values alignment.
       #Section Review report on condensed consolidated interim financial information
       #Section Condensed consolidated interim statement of financial position
       #Section Condensed consolidated interim statement of profit or loss
       #Section Condensed consolidated interim statement of comprehensive income
       #Section Condensed consolidated interim statement of changes in equity
       #Section Condensed consolidated interim statement of cash flows
       #Section Notes to the condensed consolidated interim financial information
       #Section Legal status and principal activities
       #Section Basis of preparation
       #Section Statement of compliance
       #Section Material accounting policies
       #Section Basis of consolidation
       #Section Interest Rate Benchmark Reform
       #Section Cash and balances with central banks
       #Section Investments at fair value through profit or loss
       #Section Loans, advances and Islamic financing
       #Section Non trading investment securities
       #Section Investment in associates
       #Section Investment properties
       #Section Intangibles
       #Section Due to banks and financial institutions
       #Section Commercial paper
       #Section Customer accounts and other deposits
       #Section Term borrowings
       #Section Subordinated notes
       #Section Capital and reserves
       #Section Tier 1 capital notes
       #Section Share based payment
       #Section Net foreign exchange gain
       #Section Net gain on investments and derivatives
       #Section General, administration and other operating expenses
       #Section Net impairment charge
       #Section Earnings per share
       #Section Cash and cash equivalents
       #Section Commitments and contingencies
       #Section Segmental information
       #Section Related parties
       #Section Financial risk management
       #Section Financial assets and liabilities
       #Section Comparative figures
       #Section Proposed transaction
       #Section Subsequent events
    
    4. PAGE NUMBER TRACKING:
       - Include #Section [SECTION NAME] #Page [NUMBER] at the beginning of each section
       - Track page numbers for all content
       - Note when content spans multiple pages and continued sections
    
    5. TABLE PROCESSING:
       - some of the pages are images of tables
       - Use OCR to extract text from image-based tables
       - maintain table structure and alignment
       - ensure numerical accuracy when extracting from images
       - Convert all tables to markdown format
       - Preserve exact numerical values and alignment
       - Include table headers and footnotes
       - Maintain AED million units
    
    OUTPUT FORMAT: 
    #Section [SECTION_NAME]
    #Page [PAGE_NUMBER]
    [Content...]
    
    Return as markdown with clear section headers, page numbers, and structured tables.
    """

    response = model.generate_content([prompt, document])
    return response.text


def process_earnings_presentation(pdf_path):
    """
    Process FAB Earnings Presentation PDF using Gemini
    Handles multiple filename formats:
    - FAB-Earnings-Presentation-{Quarter_number}-{year}.pdf
    - FAB-{Quarter_number}{year_in_two_digits}-Earnings-Presentation.pdf  
    - FAB-{Quarter_number}-{year_in_four_digits}-Earnings-Presentation.pdf
    """
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Please set it in the environment or in a .env file")
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    document = genai.upload_file(pdf_path)
    
    prompt = """
    You are processing a First Abu Dhabi Bank (FAB) Earnings Presentation PDF.
    
    EXTRACTION REQUIREMENTS:
    
    1. METADATA EXTRACTION:
       - Extract quarter (Q1, Q2, Q3, Q4) and year from filename/content
       - Document type: "earnings_presentation"
       - Presentation date
       - Extract from content: "Q1'25", "Q2 2024", etc. patterns
    
    2. SLIDE-BASED PROCESSING:
       - Each major slide should start with #Section [SLIDE_TITLE_OR_HEADER]
       - Include page number for each slide: #Page [PAGE_NUMBER]
       - Extract page headers for metadata enrichment
       - Focus on extracting numerical data from charts and graphics
       - Convert visual data into structured representations
       - IMPORTANT: Mention chart type (bar, line, pie) when extracting data
    
    3. KEY CONTENT TYPES:
       - Financial highlights and key metrics
       - Charts and graphs (extract data points, trends, percentages)
       - Tables with financial performance data
       - Management commentary and bullet points
       - Strategic initiatives and outlook
       - Business segment performance
       - Asset quality metrics
       - Capital and liquidity ratios
    
    4. CHART DATA EXTRACTION:
       - IMPORTANT: Mention chart type (bar, line, pie) when extracting data
       - For bar charts: extract categories and values (e.g., "Q1'24: 4.8Bn, Q1'25: 5.1Bn")
       - For line charts: extract data points over time (e.g., "NIM: 1.92%, 1.96%, 1.89%, 1.93%, 1.97%")
       - For pie charts: extract segments and percentages
       - Must Include chart titles and axis labels
       - Extract trend arrows (↑, ↓) and their meanings
    
    5. PAGE NUMBER TRACKING:
       - Include #Page [NUMBER] for each slide/page
       - Track the sequential page numbers
       - Note page headers/footers that indicate slide numbers
    
    6. TABLE PROCESSING:
       - Convert all tables to markdown format
       - Extract key financial ratios and metrics
       - Preserve quarter-over-quarter and year-over-year comparisons
       - Maintain AED million units
       - Extract percentage changes and growth rates
    
    7. COMMON SECTIONS TO EXPECT:
       #Section Key highlights
       #Section Financial review
       #Section Business segment performance
       #Section Asset quality
       #Section Capital and liquidity
       #Section Strategic update
       #Section Outlook and guidance
       #Section ESG performance
       #Section Digital transformation
       #Section Regional ESG pacesetter
       #Section Diversified growth engine
       #Section Expanding international reach
       #Section Digital & AI-led transformation
    
    8. SPECIFIC METRICS TO EXTRACT:
       - PBT (Profit Before Tax), NPAT (Net Profit After Tax)
       - RoTE (Return on Tangible Equity)
       - NIM (Net Interest Margin)
       - Cost-to-Income ratio
       - Cost of Risk
       - NPL (Non-Performing Loans) ratio
       - CET1 ratio
       - Loan growth, Deposit growth
       - Total Assets
       - Operating income breakdown
    
    9. FORMATTING FOR FINANCIAL DATA:
       - Extract numbers with their units (AED Bn, AED Mn, %)
       - Preserve growth indicators (+22% yoy, +32% qoq)
       - Extract comparative figures (Q1'24 vs Q1'25)
    
    OUTPUT FORMAT: 
    #Section [SLIDE_TITLE]
    #Page [PAGE_NUMBER]
    [Content including extracted chart data, tables, and commentary]
    
    Return as markdown with slide-based sections, page numbers, and structured data representations.
    Focus on converting visual financial data into analyzable text format.
    """
    
    response = model.generate_content([prompt, document])
    return response.text

def process_results_call(pdf_path):
    """
    Process FAB Results Call Transcript PDF using Gemini
    Format: FAB-Q{Quarter_number}-{year}-Results-Call.pdf
    """
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Please set it in the environment or in a .env file")
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    document = genai.upload_file(pdf_path)
    
    prompt = """
    You are processing a First Abu Dhabi Bank (FAB) Earnings Call Transcript PDF.

    EXTRACTION REQUIREMENTS:

    1. METADATA EXTRACTION:
       - Extract quarter (Q1, Q2, Q3, Q4) and year from filename/content
       - Document type: "results_call" if it's a results call transcript
       - Call date and time
       - List of all participants/speakers found in the document

    2. DIALOGUE PROCESSING:
       - Extract speaker names dynamically based on the pattern "Name:"
       - Use speaker names as section identifiers: #Section [EXACT_SPEAKER_NAME]
       - Include page number for each speaker turn: #Page [PAGE_NUMBER]
       - Preserve the conversational flow and extract full dialogues

    3. Dialogue content:
       - Extract each speaker's full dialogue under their section

    OUTPUT FORMAT: 
    #Section [SPEAKER_NAME]
    #Page [PAGE_NUMBER]
    [Content of speaker's dialogue the same way written in the document]
    
    Example:
    #Section Lars Kramer
    #Page 2
    [Content of Lars Kramer's speech]

    """
    
    response = model.generate_content([prompt, document])
    return response.text

def extract_quarter_year_from_filename(filename):
    """
    Extract quarter and year from various FAB earnings presentation filename formats
    """
    filename_upper = filename.upper()
    
    # Pattern 1: FAB-Q123-Earnings-Presentation.pdf (Q1 2023)
    match1 = re.search(r'FAB-Q([1-4])(\d{2})-EARNINGS-PRESENTATION', filename_upper)
    if match1:
        quarter = f"Q{match1.group(1)}"
        year = f"20{match1.group(2)}"
        return quarter, year
    
    # Pattern 2: FAB-Q1-2025-Earnings-Presentation.pdf
    match2 = re.search(r'FAB-Q([1-4])-(\d{4})-EARNINGS-PRESENTATION', filename_upper)
    if match2:
        quarter = f"Q{match2.group(1)}"
        year = match2.group(2)
        return quarter, year
    
    # Pattern 3: FAB-Earnings-Presentation-Q1-2025.pdf
    match3 = re.search(r'FAB-EARNINGS-PRESENTATION-Q([1-4])-(\d{4})', filename_upper)
    if match3:
        quarter = f"Q{match3.group(1)}"
        year = match3.group(2)
        return quarter, year
    
    # Pattern 4: FAB-Earnings-Presentation-{Quarter_number}-{year}.pdf
    match4 = re.search(r'FAB-EARNINGS-PRESENTATION-([Qq][1-4])-(\d{4})', filename_upper)
    if match4:
        quarter = match4.group(1).upper()
        year = match4.group(2)
        return quarter, year
    
    # Pattern 5: Try to extract from content patterns like "Q1'25" in the filename
    match5 = re.search(r'Q([1-4])[\'\-_]?(\d{2,4})', filename_upper)
    if match5:
        quarter = f"Q{match5.group(1)}"
        year_str = match5.group(2)
        if len(year_str) == 2:
            year = f"20{year_str}"
        else:
            year = year_str
        return quarter, year
    
    return None, None


def extract_metadata_from_content(content, doc_type, filename):
    """
    Extract standardized metadata from processed content including page numbers
    """
    metadata = {
        'document_type': doc_type,
        'filename': filename,
        'sections': [],
        'page_references': {}
    }
    
    # First try to extract from filename
    quarter, year = extract_quarter_year_from_filename(filename)
    if quarter:
        metadata['quarter'] = quarter
    if year:
        metadata['year'] = year
    
    # Also try to extract from content for earnings presentations
    if doc_type == 'earnings_presentation':
        content_upper = content.upper()
        # Look for patterns like "Q1'25", "Q1 2025", "Q125" in content
        quarter_year_patterns = [
            r'Q([1-4])[\'\\-\\s]?(\d{2})',  # Q1'25, Q1-25, Q1 25
            r'Q([1-4])\\s+(20\d{2})',       # Q1 2025
            r'Q([1-4])(\d{2})(?!\\d)'       # Q125 (but not Q1250)
        ]
        
        for pattern in quarter_year_patterns:
            match = re.search(pattern, content_upper)
            if match:
                found_quarter = f"Q{match.group(1)}"
                found_year = match.group(2)
                if len(found_year) == 2:
                    found_year = f"20{found_year}"
                
                if not metadata.get('quarter'):
                    metadata['quarter'] = found_quarter
                if not metadata.get('year'):
                    metadata['year'] = found_year
                break
    
    if metadata.get('quarter') and metadata.get('year'):
        metadata['fiscal_period'] = f"{metadata['year']}-{metadata['quarter']}"
    
    # Extract sections and their page numbers
    section_pattern = r'#Section\s+([^\n]+)\s+#Page\s+(\d+)'
    section_matches = re.findall(section_pattern, content)
    
    sections_with_pages = []
    for section_name, page_num in section_matches:
        # Clean up section name (remove any trailing dots or spaces)
        clean_section_name = section_name.strip().rstrip('. ')
        sections_with_pages.append({
            'section': clean_section_name,
            'page': int(page_num)
        })
        # Build page reference dictionary
        if clean_section_name not in metadata['page_references']:
            metadata['page_references'][clean_section_name] = []
        metadata['page_references'][clean_section_name].append(int(page_num))
    
    metadata['sections'] = sections_with_pages
    
    # For results calls, extract speaker names specifically
    if doc_type == 'results_call':
        # Filter out financial statement sections to get only speaker names
        financial_sections = [
            'Review report on condensed consolidated interim financial information',
            'Condensed consolidated interim statement of financial position',
            # ... (all the financial statement sections)
        ]
        
        speaker_names = [section['section'] for section in sections_with_pages 
                        if section['section'] not in financial_sections]
        metadata['speakers'] = list(set(speaker_names))
    
    # Extract all unique pages mentioned
    all_pages = set()
    for section_data in sections_with_pages:
        all_pages.add(section_data['page'])
    metadata['total_pages'] = max(all_pages) if all_pages else 0
    metadata['pages_covered'] = sorted(list(all_pages))
    
    return metadata

def process_fab_document(pdf_path):
    """
    Main function to process any FAB document using Gemini with page number tracking
    """
    filename = pdf_path.split('/')[-1]
    filename_upper = filename.upper()
    
    # Determine document type with strict pattern matching (order matters - most specific first)
    
    # 1. Check for Financial Statements (most specific pattern)
    if 'FS-' in filename_upper:
        doc_type = 'financial_statement'
        print("Processing as Financial Statement")
        content = process_financial_statement(pdf_path)
    
    # 2. Check for Results Calls (before earnings presentation to avoid false matches)
    elif 'RESULTS-CALL' in filename_upper or 'RESULTS CALL' in filename_upper:
        doc_type = 'results_call'
        print("Processing as Results Call")
        content = process_results_call(pdf_path)
    
    # 3. Check for Earnings Presentations (specific keywords required)
    elif any(keyword in filename_upper for keyword in ['EARNINGS-PRESENTATION', 'EARNINGS PRESENTATION']) or \
         (filename_upper.endswith('-PRESENTATION.PDF') and 'FS-' not in filename_upper and 'CALL' not in filename_upper):
        doc_type = 'earnings_presentation'
        print("Processing as Earnings Presentation")
        content = process_earnings_presentation(pdf_path)
    
    else:
        raise ValueError(f"Unknown document type for file: {filename}. Expected 'FS-', 'Results-Call', or 'Earnings-Presentation' in filename.")
    
    # Extract metadata including page numbers
    metadata = extract_metadata_from_content(content, doc_type, filename)
    
    # For earnings presentations, try to extract quarter/year from filename if not found in content
    if doc_type == 'earnings_presentation' and (not metadata.get('quarter') or not metadata.get('year')):
        quarter, year = extract_quarter_year_from_filename(filename)
        if quarter and year:
            metadata['quarter'] = quarter
            metadata['year'] = year
            if quarter and year:
                metadata['fiscal_period'] = f"{year}-{quarter}"
    
    return {
        'content': content,
        'metadata': metadata,
        'document_type': doc_type
    }