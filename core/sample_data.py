"""
Sample Data for Testing Handai Tools
Diverse datasets for qualitative analysis, coding, and data generation
"""

from typing import Dict, List, Any


# ==========================================
# QUALITATIVE CODER SAMPLES
# ==========================================

# Product Reviews - Sentiment Analysis
PRODUCT_REVIEWS = {
    "review_id": [f"REV{i:04d}" for i in range(1, 21)],
    "text": [
        "I absolutely love this product! Best purchase ever. The quality exceeded my expectations.",
        "Terrible experience. Would not recommend to anyone. Broke after two days of use.",
        "It's okay, nothing special but does the job. Average quality for the price.",
        "Amazing quality and fast shipping. Very happy with my purchase!",
        "Broke after one week. Complete waste of money. Customer service was unhelpful.",
        "Decent value for the price. Could be better but no major complaints.",
        "This exceeded all my expectations! Fantastic product, will buy again.",
        "Not worth it. Save your money and buy something else.",
        "Pretty good overall. Minor issues but satisfied with the purchase.",
        "Outstanding service and product. Five stars all the way!",
        "Mixed feelings. Some features are great, others are disappointing.",
        "Worst purchase I've made this year. Total disappointment.",
        "Surprisingly good for such a low price. Pleasantly surprised.",
        "The product is fine but the packaging was damaged when it arrived.",
        "Perfect for my needs. Exactly what I was looking for.",
        "Returned it immediately. Did not match the description at all.",
        "Good quality but shipping took forever. Two weeks to arrive.",
        "Love it! Already recommended to all my friends and family.",
        "Mediocre at best. There are better options out there.",
        "Exceptional craftsmanship. You can tell this was made with care.",
    ],
    "category": [
        "Electronics", "Clothing", "Home", "Electronics", "Toys",
        "Home", "Beauty", "Electronics", "Clothing", "Food",
        "Electronics", "Home", "Kitchen", "Books", "Sports",
        "Clothing", "Electronics", "Beauty", "Home", "Kitchen"
    ],
    "rating": [5, 1, 3, 5, 1, 3, 5, 1, 4, 5, 3, 1, 4, 3, 5, 1, 3, 5, 2, 5],
    "date": [
        "2024-01-15", "2024-01-14", "2024-01-13", "2024-01-12", "2024-01-11",
        "2024-01-10", "2024-01-09", "2024-01-08", "2024-01-07", "2024-01-06",
        "2024-01-05", "2024-01-04", "2024-01-03", "2024-01-02", "2024-01-01",
        "2023-12-31", "2023-12-30", "2023-12-29", "2023-12-28", "2023-12-27"
    ]
}

# Interview Transcripts - Healthcare Worker Experience
HEALTHCARE_INTERVIEWS = {
    "participant_id": [f"HCW{i:03d}" for i in range(1, 16)],
    "role": [
        "Nurse", "Doctor", "Nurse", "Admin", "Doctor",
        "Nurse", "Technician", "Doctor", "Nurse", "Admin",
        "Technician", "Nurse", "Doctor", "Nurse", "Admin"
    ],
    "experience_years": [5, 12, 3, 8, 20, 7, 4, 15, 2, 10, 6, 9, 18, 1, 14],
    "response": [
        "The pandemic changed everything. We went from a normal routine to crisis mode overnight. I've never felt so exhausted yet so purposeful in my career.",
        "Burnout is real and it's everywhere. I see colleagues leaving the profession weekly. The system needs fundamental change, not just appreciation posts.",
        "What keeps me going is the patients. When someone recovers and thanks you, it reminds you why you chose this path despite all the challenges.",
        "The administrative burden has tripled. For every hour of patient care, there's two hours of paperwork. It's unsustainable.",
        "I've been doing this for twenty years and I've never seen morale this low. We need better staffing ratios and mental health support.",
        "The team camaraderie is what saves us. We lean on each other when things get tough. Without that support system, I wouldn't survive.",
        "Technology has been both a blessing and a curse. New systems help but the constant updates and training requirements add to our stress.",
        "I worry about the next generation of healthcare workers. Who will want to enter this field after seeing what we've been through?",
        "Starting my career during this time has been intense. I feel like I've gained five years of experience in two years.",
        "Resource allocation is my biggest challenge. Making decisions about who gets what when there isn't enough is soul-crushing.",
        "The equipment issues are frustrating. We're asked to do more with less while being held to higher standards.",
        "Night shifts are the hardest. The loneliness combined with high-acuity patients takes a toll on your mental health.",
        "Leadership needs to listen more. The decisions made in boardrooms don't always reflect the reality on the floor.",
        "I'm proud of what we've accomplished but exhausted by what it took to get here. Recovery will take years.",
        "Communication between departments has improved but there's still so much siloing. We need to work better together.",
    ]
}

# Customer Support Tickets - Issue Classification
SUPPORT_TICKETS = {
    "ticket_id": [f"TKT{i:05d}" for i in range(1, 21)],
    "subject": [
        "Can't login to my account",
        "Refund request for order #12345",
        "Product arrived damaged",
        "How do I change my password?",
        "Billing discrepancy on my statement",
        "Feature request: dark mode",
        "App crashes on startup",
        "Shipping taking too long",
        "Need invoice for business expense",
        "Unauthorized charge on my card",
        "How to export my data?",
        "Subscription cancellation",
        "Missing items in my order",
        "Can't connect to Bluetooth",
        "Account was hacked",
        "Want to upgrade my plan",
        "Product not as described",
        "Need technical documentation",
        "Delivery address change",
        "Promotional code not working",
    ],
    "description": [
        "I've tried resetting my password three times but I still can't access my account. The reset email never arrives.",
        "I would like a full refund for order #12345. The product quality was much lower than advertised and I'm disappointed.",
        "My package arrived today but the box was crushed and the item inside is broken. I need a replacement sent immediately.",
        "Where can I find the option to change my password? I've looked everywhere in settings but can't find it.",
        "My last statement shows a charge of $99 but my plan is only $49/month. Please explain this discrepancy.",
        "Please add a dark mode option to the app. Using it at night is too bright and strains my eyes.",
        "Every time I open the app it immediately closes. I've tried reinstalling but the problem persists. iPhone 14, iOS 17.",
        "I ordered 10 days ago and my package still shows as 'processing'. When will it actually ship?",
        "I need an itemized invoice for this purchase for my business accounting. Can you email it to me?",
        "I see a $200 charge from your company that I didn't make. I need this reversed immediately and my account secured.",
        "I'm switching to a competitor and need to export all my data. What's the process for getting my information?",
        "Please cancel my subscription effective immediately. I no longer need this service.",
        "Order #67890 arrived but was missing 2 of the 5 items. I received only the small items, not the main products.",
        "The Bluetooth won't pair with my phone. I've tried everything in the troubleshooting guide.",
        "Someone accessed my account and changed my email. I can't get back in and I'm worried about my saved payment info.",
        "I'd like to upgrade from Basic to Premium. What's the price difference and how do I make the switch?",
        "The color in the photos looked blue but what I received is clearly purple. This is false advertising.",
        "I'm a developer integrating your API. Where can I find the technical documentation and code examples?",
        "I realized I put the wrong shipping address. Can you update it to 123 New St before it ships?",
        "The code SAVE20 isn't applying any discount at checkout even though it should be valid until next week.",
    ],
    "priority": [
        "high", "medium", "high", "low", "high",
        "low", "high", "medium", "low", "critical",
        "medium", "low", "medium", "medium", "critical",
        "low", "medium", "low", "medium", "low"
    ],
    "channel": [
        "email", "chat", "phone", "chat", "email",
        "app", "email", "chat", "email", "phone",
        "email", "chat", "phone", "chat", "phone",
        "chat", "email", "email", "chat", "app"
    ]
}


# ==========================================
# CODEBOOK GENERATOR SAMPLES
# ==========================================

# Student Learning Experience (already exists, enhanced version)
LEARNING_EXPERIENCE = {
    "participant_id": [f"STU{i:03d}" for i in range(1, 21)],
    "response": [
        "The online learning platform made it easy to access materials anytime. I appreciated the flexibility but sometimes felt isolated from other students.",
        "I found the group discussions very engaging. Hearing different perspectives helped me understand the material better, though some sessions ran too long.",
        "The professor's feedback was always constructive and timely. I felt supported throughout the course even when struggling with difficult concepts.",
        "Technical issues with the video conferencing were frustrating. When the technology worked, the experience was good, but reliability was a problem.",
        "I loved the hands-on projects. They helped me apply what I learned to real-world situations. The theory lectures were less engaging.",
        "The workload was overwhelming at times. I wish there was better balance between readings, assignments, and exams throughout the semester.",
        "Office hours were very helpful for clarifying confusing topics. The professor was approachable and took time to explain things multiple ways.",
        "Collaboration with classmates was the highlight. We formed study groups that continued beyond the course. The social aspect enhanced learning.",
        "The course materials were outdated in some areas. I had to supplement with external resources to get current industry perspectives.",
        "Self-paced modules allowed me to learn at my own speed. However, I sometimes procrastinated without regular deadlines pushing me forward.",
        "The assessment methods seemed fair and aligned with learning objectives. I appreciated having multiple ways to demonstrate my understanding.",
        "Navigating the learning management system was confusing initially. Once I figured it out, accessing resources became straightforward.",
        "Guest speakers from industry provided valuable real-world insights. These sessions were among the most memorable parts of the course.",
        "I struggled with the lack of immediate feedback on assignments. Waiting weeks for grades made it hard to know if I was on track.",
        "The course built a strong foundation in the subject. I feel confident applying these skills in my future career.",
        "Discussion boards felt forced and artificial. I would have preferred more natural ways to interact with classmates.",
        "The recorded lectures were helpful for reviewing complex topics. Being able to pause and rewatch was essential for my learning.",
        "I felt anxious about participating in live sessions. The pressure of speaking in front of everyone was intimidating.",
        "The course exceeded my expectations. I came in skeptical about online learning but was pleasantly surprised by the quality.",
        "Time zone differences made synchronous sessions difficult. More asynchronous options would have been appreciated.",
    ],
    "course_format": [
        "online", "hybrid", "online", "hybrid", "in-person",
        "online", "in-person", "hybrid", "online", "online",
        "in-person", "online", "hybrid", "online", "in-person",
        "online", "online", "hybrid", "online", "online"
    ],
    "satisfaction_score": [4, 5, 5, 2, 4, 2, 5, 5, 3, 3, 4, 3, 5, 2, 5, 2, 4, 2, 5, 3]
}

# Employee Exit Interviews
EXIT_INTERVIEWS = {
    "employee_id": [f"EMP{i:04d}" for i in range(1, 16)],
    "department": [
        "Engineering", "Sales", "Marketing", "Engineering", "HR",
        "Finance", "Engineering", "Sales", "Operations", "Marketing",
        "Engineering", "Customer Success", "Product", "Engineering", "Sales"
    ],
    "tenure_years": [3.5, 1.2, 5.0, 2.8, 4.2, 6.1, 1.8, 3.3, 2.5, 4.8, 0.9, 3.7, 2.1, 5.5, 1.5],
    "reason_for_leaving": [
        "I received an offer I couldn't refuse - 40% salary increase and a senior title. I loved working here but had to think about my family's financial future.",
        "The lack of career progression was frustrating. I've been in the same role for over a year with no clear path forward despite multiple conversations.",
        "Management changes created a toxic environment. The new leadership has different values than what attracted me to this company originally.",
        "I'm relocating for personal reasons. The company has been great but they don't have remote options for my role.",
        "Burnout. The constant pressure to do more with less finally caught up with me. I need time to recover before starting somewhere new.",
        "Better opportunity aligned with my long-term career goals. This role taught me a lot but I've outgrown what I can learn here.",
        "The tech stack is outdated and there's no willingness to modernize. As an engineer, working with legacy systems is career-limiting.",
        "Commission structure changed unfairly. My earning potential dropped 30% even though I'm hitting the same targets.",
        "Work-life balance was non-existent. Being expected to be available 24/7 is not sustainable or healthy.",
        "I didn't feel my contributions were valued. Despite successful campaigns, credit always went elsewhere.",
        "The job wasn't what was described in the interview. Day-to-day responsibilities are completely different from what I signed up for.",
        "Customer success isn't a priority here despite what leadership says. I couldn't keep making promises we wouldn't keep.",
        "Product direction keeps changing. It's impossible to build anything meaningful when strategy shifts every quarter.",
        "Found a startup opportunity that's too exciting to pass up. I'll miss the stability but the challenge is calling.",
        "Honestly, the compensation isn't competitive anymore. Cost of living has increased but salaries haven't kept pace.",
    ]
}


# ==========================================
# AUTOMATOR SAMPLES
# ==========================================

# News Articles - Classification
NEWS_ARTICLES = {
    "article_id": [f"NEWS{i:04d}" for i in range(1, 21)],
    "headline": [
        "Tech Giants Report Record Quarterly Earnings",
        "Climate Summit Reaches Historic Agreement on Emissions",
        "Local Team Wins Championship After 30-Year Drought",
        "New Study Links Sleep Quality to Cognitive Performance",
        "Stock Market Reaches All-Time High Amid Economic Optimism",
        "Revolutionary Cancer Treatment Shows Promise in Trials",
        "City Council Approves Affordable Housing Development",
        "Famous Actor Announces Retirement from Film Industry",
        "Scientists Discover New Species in Deep Ocean Expedition",
        "Central Bank Raises Interest Rates to Combat Inflation",
        "Electric Vehicle Sales Surpass Traditional Cars for First Time",
        "Music Festival Breaks Attendance Records This Weekend",
        "Earthquake Preparedness Campaign Launches Nationwide",
        "Startup Raises $100M to Transform Food Delivery",
        "Olympic Committee Announces Host City for 2036 Games",
        "Healthcare Workers Strike Over Working Conditions",
        "Artificial Intelligence System Beats Human Experts at Diagnosis",
        "Real Estate Market Shows Signs of Cooling Down",
        "Children's Author Wins Prestigious Literary Award",
        "Space Agency Reveals Plans for Mars Colony by 2040",
    ],
    "content": [
        "Major technology companies including Apple, Google, and Microsoft all reported earnings that exceeded analyst expectations. The strong results were driven by cloud computing services and advertising revenue. Investors responded positively, pushing stock prices higher in after-hours trading.",
        "After two weeks of intense negotiations, world leaders agreed to a binding framework to reduce carbon emissions by 50% by 2035. Environmental groups cautiously welcomed the agreement while noting implementation challenges ahead.",
        "The underdog team clinched the title in a dramatic overtime victory, ending three decades of championship futility. Fans flooded the streets in celebration as players embraced on the field.",
        "Researchers found that individuals who consistently get 7-9 hours of quality sleep perform significantly better on memory and problem-solving tasks. The study followed 5,000 participants over five years.",
        "Investor confidence pushed major indices to unprecedented levels as employment data and corporate profits continued to exceed expectations. Analysts remain cautiously optimistic about sustained growth.",
        "Phase 3 clinical trials of the novel immunotherapy treatment showed a 70% response rate in patients with advanced-stage disease. Researchers say approval could come within 18 months.",
        "The 500-unit development will reserve 30% of apartments for below-market-rate tenants. Community members expressed mixed reactions at the public hearing.",
        "The acclaimed performer cited a desire to spend more time with family after a 40-year career that included three Academy Awards and dozens of memorable roles.",
        "Marine biologists identified three new species of fish living near hydrothermal vents at depths exceeding 3,000 meters. The discovery challenges assumptions about life in extreme environments.",
        "The quarter-point increase marks the sixth consecutive rate hike as policymakers attempt to bring inflation back to target levels without triggering a recession.",
        "Industry data shows EVs accounted for 51% of new car sales last month, a milestone that experts predicted wouldn't arrive until 2027. Government incentives and improved charging infrastructure contributed to the shift.",
        "Over 200,000 fans attended the three-day event featuring more than 100 artists across eight stages. Organizers called it the most successful festival in the event's 15-year history.",
        "Officials are distributing emergency kits and conducting community drills in preparation for potential seismic activity. Geologists have warned of increased risk along the fault line.",
        "The funding round was led by top-tier venture capital firms betting on the company's AI-powered logistics platform. The startup plans to expand to 50 new cities within 12 months.",
        "The selection committee chose the bid from Mumbai, India, citing the city's modern infrastructure and passionate sporting culture. It will be the first time the subcontinent hosts the Summer Games.",
        "Thousands of nurses and support staff walked off the job demanding better pay, safer staffing ratios, and improved mental health resources. Hospital administrators said contingency plans are in place.",
        "The deep learning system achieved 94% accuracy in identifying skin cancers from images, outperforming the 87% average among board-certified dermatologists in the controlled study.",
        "Home prices fell for the third consecutive month as rising mortgage rates reduced buyer purchasing power. Some economists see this as a healthy correction after years of unsustainable growth.",
        "The beloved author known for imaginative worlds and diverse characters will receive the award at a ceremony next month. Her books have been translated into 40 languages.",
        "The ambitious plan includes establishing a permanent human presence on the red planet within 20 years. Private sector partnerships will be essential to meeting the timeline and budget.",
    ],
    "source": [
        "Financial Times", "Reuters", "Sports Daily", "Science Weekly", "Bloomberg",
        "Medical Journal", "Local Gazette", "Entertainment Tonight", "Nature News", "Wall Street Journal",
        "Auto Industry Weekly", "Music Magazine", "Safety First", "Tech Crunch", "Olympic Channel",
        "Health Workers Union", "AI Research Quarterly", "Real Estate Report", "Book Review", "Space News"
    ],
    "date": [
        "2024-01-20", "2024-01-19", "2024-01-18", "2024-01-17", "2024-01-16",
        "2024-01-15", "2024-01-14", "2024-01-13", "2024-01-12", "2024-01-11",
        "2024-01-10", "2024-01-09", "2024-01-08", "2024-01-07", "2024-01-06",
        "2024-01-05", "2024-01-04", "2024-01-03", "2024-01-02", "2024-01-01"
    ]
}

# Social Media Posts - Content Moderation
SOCIAL_MEDIA_POSTS = {
    "post_id": [f"POST{i:06d}" for i in range(1, 21)],
    "content": [
        "Just finished my morning run! 5K in under 25 minutes. Feeling great!",
        "This product is a SCAM!!! DO NOT BUY!!! They stole my money!!!",
        "Happy birthday to my amazing mom! Love you so much!",
        "Anyone else think the new policy is ridiculous? Share if you agree!",
        "Check out my new blog post about sustainable living: [link]",
        "I can't believe what I just saw on the news. This world is crazy.",
        "FREE iPhone giveaway!!! Click here to claim yours NOW!!!",
        "Grateful for another day. Remember to tell your loved ones you appreciate them.",
        "This politician should be [removed for violence]. I'm so angry right now.",
        "My cat did the funniest thing today. Video in comments!",
        "BREAKING: Major announcement coming soon. Stay tuned!",
        "I respectfully disagree with the previous comment. Here's my perspective...",
        "Buy my course and make $10,000/month working from home! DM for details.",
        "Beautiful sunset from my balcony tonight. Nature is amazing.",
        "If you don't support [cause], you're part of the problem.",
        "Just got promoted at work! Hard work pays off!",
        "This restaurant gave me food poisoning. Avoid at all costs. @healthdept",
        "Meditation has changed my life. 10 minutes a day makes a huge difference.",
        "Why is everyone so sensitive these days? Can't say anything anymore.",
        "Throwback to our vacation last summer. Can't wait to travel again!",
    ],
    "username": [
        "fitness_jane", "angry_customer42", "family_first", "political_pete", "eco_emma",
        "news_watcher", "totally_not_spam", "grateful_gary", "heated_henry", "cat_mom_99",
        "hype_account", "respectful_debater", "get_rich_quick", "nature_lover", "activist_ally",
        "career_climber", "honest_reviewer", "mindful_mary", "traditional_tom", "travel_bug"
    ],
    "likes": [45, 892, 234, 1205, 67, 23, 5, 189, 3420, 567, 12, 89, 3, 445, 2100, 312, 678, 156, 890, 223],
    "timestamp": [
        "2024-01-20 07:30:00", "2024-01-20 09:15:00", "2024-01-20 10:00:00", "2024-01-20 11:45:00",
        "2024-01-20 12:30:00", "2024-01-20 14:00:00", "2024-01-20 15:15:00", "2024-01-20 16:30:00",
        "2024-01-20 17:45:00", "2024-01-20 18:00:00", "2024-01-20 19:30:00", "2024-01-20 20:00:00",
        "2024-01-20 20:30:00", "2024-01-20 21:00:00", "2024-01-20 21:30:00", "2024-01-20 22:00:00",
        "2024-01-20 22:30:00", "2024-01-20 23:00:00", "2024-01-20 23:30:00", "2024-01-21 00:00:00"
    ]
}


# ==========================================
# CONSENSUS CODER SAMPLES
# ==========================================

# Research Paper Abstracts - Multi-label Classification
RESEARCH_ABSTRACTS = {
    "paper_id": [f"PAPER{i:04d}" for i in range(1, 16)],
    "title": [
        "Deep Learning Approaches for Medical Image Segmentation",
        "The Impact of Remote Work on Employee Wellbeing",
        "Sustainable Agriculture Practices in Developing Nations",
        "Quantum Computing Applications in Cryptography",
        "Social Media's Influence on Political Polarization",
        "Gene Therapy Advances for Rare Diseases",
        "Machine Learning for Climate Change Prediction",
        "Urban Planning Strategies for Smart Cities",
        "Blockchain Technology in Supply Chain Management",
        "Psychological Effects of Prolonged Social Isolation",
        "Renewable Energy Storage Solutions",
        "Natural Language Processing for Legal Documents",
        "Biodiversity Conservation in Tropical Rainforests",
        "Autonomous Vehicle Safety Standards",
        "Digital Literacy in Elderly Populations",
    ],
    "abstract": [
        "This paper presents a novel convolutional neural network architecture for automated segmentation of tumors in MRI scans. Our approach achieves 95% accuracy on benchmark datasets, outperforming existing methods by 8%. Clinical validation with radiologists confirms the practical utility of our system for diagnostic support.",
        "We surveyed 2,500 remote workers across 15 countries to understand how working from home affects mental health, productivity, and work-life balance. Results indicate improved flexibility but increased feelings of isolation. We propose evidence-based guidelines for organizations implementing hybrid work policies.",
        "This longitudinal study examines the adoption of sustainable farming techniques in Sub-Saharan Africa over a 10-year period. We find that farmer cooperatives and mobile technology access are key predictors of successful implementation. Policy recommendations for agricultural development are discussed.",
        "We demonstrate a quantum algorithm that can break RSA-2048 encryption in polynomial time on a fault-tolerant quantum computer. While such computers don't yet exist, our findings underscore the urgent need for post-quantum cryptographic standards. We analyze migration pathways for critical infrastructure.",
        "Using computational text analysis of 50 million social media posts, we trace the evolution of political discourse from 2010-2023. Our findings reveal that algorithmic amplification of emotionally charged content significantly contributes to ideological clustering and reduced cross-partisan engagement.",
        "We report successful Phase 2 clinical trial results for a CRISPR-based therapy targeting Duchenne muscular dystrophy. Treatment resulted in sustained dystrophin expression in 80% of patients with minimal adverse effects. This represents a major milestone in precision medicine for genetic disorders.",
        "This paper introduces an ensemble deep learning model that integrates satellite imagery, ocean temperature data, and atmospheric measurements to predict extreme weather events. Our system provides 30% earlier warning compared to traditional meteorological models.",
        "We analyze smart city implementations across 25 municipalities to identify best practices in urban technology deployment. Key success factors include citizen engagement, data privacy frameworks, and interoperability standards. Failed initiatives share common pitfalls that future planners should avoid.",
        "We present a permissioned blockchain solution that enables end-to-end supply chain transparency while preserving commercial confidentiality. Pilot implementation with a major retailer reduced counterfeit incidents by 60% and improved recall response time from weeks to hours.",
        "This meta-analysis synthesizes 89 studies on the psychological impact of isolation during the COVID-19 pandemic. We identify depression, anxiety, and sleep disorders as primary outcomes, with effects varying by age and living situation. Effective interventions and at-risk populations are discussed.",
        "We review recent advances in grid-scale energy storage, comparing lithium-ion, flow batteries, and hydrogen systems across cost, efficiency, and environmental dimensions. Emerging solid-state technologies show promise for meeting 2030 renewable energy targets.",
        "Our transformer-based model achieves state-of-the-art performance on contract analysis, extracting key clauses with 92% F1 score. We demonstrate practical applications in M&A due diligence, reducing legal review time by 70% while maintaining accuracy.",
        "This 15-year study documents species population changes in the Amazon basin following implementation of indigenous-led conservation programs. Areas under community management showed 40% less deforestation and maintained higher species diversity than government-managed reserves.",
        "We propose a comprehensive safety framework for Level 4 autonomous vehicles, synthesizing requirements from engineering, regulatory, and ethical perspectives. Our testing methodology addresses edge cases that current standards fail to consider adequately.",
        "This study examines barriers to technology adoption among adults over 65, finding that interface complexity and privacy concerns outweigh physical limitations. We design and validate an age-friendly digital skills curriculum that significantly improves online engagement.",
    ],
    "year": [2024, 2023, 2024, 2024, 2023, 2024, 2023, 2024, 2023, 2021, 2024, 2023, 2022, 2024, 2023]
}


# ==========================================
# MIXED LENGTH SAMPLES (for testing layout)
# ==========================================

# Mixed Feedback - Varying text lengths for testing
MIXED_FEEDBACK = {
    "id": [f"FB{i:03d}" for i in range(1, 16)],
    "feedback": [
        "Great!",
        "Not satisfied.",
        "The product works exactly as described. Happy with my purchase.",
        "OK",
        "I've been using this service for about three months now and I have to say the experience has been quite mixed. On one hand, the core functionality works well and the interface is intuitive. However, there have been several instances where the system went down during peak hours, which was frustrating. Customer support was helpful when I reached out, but the wait times were longer than expected. Overall, I think there's a lot of potential here, but some reliability improvements would make a big difference.",
        "Love it!",
        "The delivery was fast and the packaging was secure. The item itself meets my expectations.",
        "Terrible experience from start to finish. First, the website kept crashing when I tried to place my order. Then, when I finally managed to complete the purchase, the confirmation email never arrived. I had to contact support multiple times just to verify my order went through. When the product finally arrived two weeks late, it was the wrong color. The return process was equally painful - I had to pay for return shipping even though it was their mistake. I've requested a full refund and will never shop here again. Save yourself the headache and buy elsewhere.",
        "5 stars",
        "Works well for basic tasks but lacks advanced features that competitors offer.",
        "Meh.",
        "This exceeded all my expectations! I was initially skeptical because of the low price point, but the quality is surprisingly good. The material feels durable, the stitching is solid, and it looks exactly like the pictures. I've already recommended it to several friends and family members. Will definitely be purchasing more items from this seller in the future.",
        "Average product, average price, average experience. Nothing special but nothing wrong either.",
        "NO",
        "After extensive research comparing multiple options, I decided to try this product based on the positive reviews. I'm pleased to report that it lives up to the hype. Setup was straightforward, performance is excellent, and the customer service team answered all my pre-purchase questions promptly and thoroughly.",
    ],
    "rating": [5, 2, 4, 3, 3, 5, 4, 1, 5, 3, 2, 5, 3, 1, 4],
    "date": [
        "2024-01-15", "2024-01-14", "2024-01-13", "2024-01-12", "2024-01-11",
        "2024-01-10", "2024-01-09", "2024-01-08", "2024-01-07", "2024-01-06",
        "2024-01-05", "2024-01-04", "2024-01-03", "2024-01-02", "2024-01-01",
    ]
}


# ==========================================
# DOCUMENT PROCESSING SAMPLES
# ==========================================

# Legal Case Summaries (English)
LEGAL_CASES = {
    "case_id": [f"CASE{i:04d}" for i in range(1, 11)],
    "case_name": [
        "Smith v. ABC Corporation",
        "State v. Johnson",
        "Williams Estate Matter",
        "Brown v. City of Springfield",
        "In re: XYZ Bankruptcy",
        "Davis v. Medical Center",
        "People v. Anderson",
        "Thompson v. Insurance Co.",
        "Martinez Custody Dispute",
        "Chen v. Tech Startup Inc.",
    ],
    "summary": [
        "Plaintiff alleges wrongful termination after reporting safety violations. Defendant claims termination was due to performance issues. Key witness testimony conflicts on timeline of events. Settlement negotiations ongoing.",
        "Defendant charged with second-degree burglary. Security footage shows individual matching description entering premises. Defense argues mistaken identity and alibi from family members.",
        "Contested will involving three adult children and charitable organization. Decedent's mental capacity at time of signing disputed. Prior will from 2019 gave equal shares to children; current will leaves 80% to charity.",
        "Class action lawsuit alleging excessive force by police department. Plaintiffs seek policy reforms and damages. City has agreed to independent review but denies systemic issues.",
        "Chapter 11 reorganization plan submitted by debtor corporation. Creditors committee objects to treatment of unsecured claims. Court-appointed examiner investigating pre-bankruptcy asset transfers.",
        "Medical malpractice claim arising from delayed cancer diagnosis. Expert testimony establishes deviation from standard of care. Defendant argues patient's non-compliance contributed to outcome.",
        "Defendant convicted of aggravated assault following bar altercation. Victim sustained serious injuries requiring hospitalization. Self-defense claim rejected by jury. Sentencing pending.",
        "Denial of homeowner's insurance claim after fire damage. Insurer alleges policy exclusion applies due to vacant property status. Policyholder claims exclusion notification was inadequate.",
        "Petition for modification of custody arrangement. Father seeks increased parenting time citing changed circumstances. Mother opposes citing stability concerns for minor children.",
        "Employment discrimination complaint alleging gender-based pay disparity. Plaintiff, a senior engineer, provides statistical analysis showing wage gap. Defendant claims differences explained by legitimate factors.",
    ],
    "case_type": [
        "Employment", "Criminal", "Probate", "Civil Rights", "Bankruptcy",
        "Medical Malpractice", "Criminal", "Insurance", "Family", "Employment"
    ],
    "status": [
        "Pending", "Trial", "Hearing Scheduled", "Discovery", "Pending Approval",
        "Settled", "Sentencing", "Appeal", "Mediation", "Investigation"
    ]
}


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_sample_data(name: str) -> Dict[str, List[Any]]:
    """
    Get sample data by name.

    Available datasets:
    - product_reviews: Product reviews with sentiment
    - healthcare_interviews: Healthcare worker interviews
    - support_tickets: Customer support tickets
    - learning_experience: Student learning experience
    - exit_interviews: Employee exit interviews
    - news_articles: News articles for classification
    - social_media_posts: Social media for moderation
    - research_abstracts: Research papers for classification
    - legal_cases: Legal case summaries
    """
    datasets = {
        "product_reviews": PRODUCT_REVIEWS,
        "healthcare_interviews": HEALTHCARE_INTERVIEWS,
        "support_tickets": SUPPORT_TICKETS,
        "learning_experience": LEARNING_EXPERIENCE,
        "exit_interviews": EXIT_INTERVIEWS,
        "news_articles": NEWS_ARTICLES,
        "social_media_posts": SOCIAL_MEDIA_POSTS,
        "research_abstracts": RESEARCH_ABSTRACTS,
        "legal_cases": LEGAL_CASES,
        "mixed_feedback": MIXED_FEEDBACK,
    }
    return datasets.get(name, {})


def get_available_datasets() -> List[str]:
    """Get list of available sample dataset names."""
    return [
        "product_reviews",
        "healthcare_interviews",
        "support_tickets",
        "learning_experience",
        "exit_interviews",
        "news_articles",
        "social_media_posts",
        "research_abstracts",
        "legal_cases",
        "mixed_feedback",
    ]


def get_dataset_info() -> Dict[str, Dict[str, str]]:
    """Get information about each available dataset."""
    return {
        "product_reviews": {
            "name": "Product Reviews",
            "description": "20 product reviews with sentiment, category, and ratings",
            "use_case": "Sentiment analysis, review classification",
            "rows": 20,
            "text_column": "text"
        },
        "healthcare_interviews": {
            "name": "Healthcare Worker Interviews",
            "description": "15 interview responses from healthcare workers about their experiences",
            "use_case": "Thematic analysis, qualitative coding",
            "rows": 15,
            "text_column": "response"
        },
        "support_tickets": {
            "name": "Customer Support Tickets",
            "description": "20 support tickets with subject, description, priority, and channel",
            "use_case": "Issue classification, priority detection",
            "rows": 20,
            "text_column": "description"
        },
        "learning_experience": {
            "name": "Student Learning Experience",
            "description": "20 student responses about online/hybrid learning",
            "use_case": "Educational research, satisfaction analysis",
            "rows": 20,
            "text_column": "response"
        },
        "exit_interviews": {
            "name": "Employee Exit Interviews",
            "description": "15 exit interview responses with reasons for leaving",
            "use_case": "HR analytics, retention analysis",
            "rows": 15,
            "text_column": "reason_for_leaving"
        },
        "news_articles": {
            "name": "News Articles",
            "description": "20 news articles with headlines, content, and sources",
            "use_case": "Topic classification, summarization",
            "rows": 20,
            "text_column": "content"
        },
        "social_media_posts": {
            "name": "Social Media Posts",
            "description": "20 social media posts for content moderation",
            "use_case": "Content moderation, spam detection",
            "rows": 20,
            "text_column": "content"
        },
        "research_abstracts": {
            "name": "Research Paper Abstracts",
            "description": "15 research paper abstracts across multiple disciplines",
            "use_case": "Multi-label classification, topic extraction",
            "rows": 15,
            "text_column": "abstract"
        },
        "legal_cases": {
            "name": "Legal Case Summaries",
            "description": "10 legal case summaries with type and status",
            "use_case": "Entity extraction, case classification",
            "rows": 10,
            "text_column": "summary"
        },
        "mixed_feedback": {
            "name": "Mixed Feedback (Varying Lengths)",
            "description": "15 feedback items with short, medium, and long texts",
            "use_case": "Testing layout with varying text lengths",
            "rows": 15,
            "text_column": "feedback"
        },
    }
