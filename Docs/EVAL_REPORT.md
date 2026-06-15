# Claims Adjudication Evaluation Report

| Case ID & Name | Scenario Description | Expected Outcome (Decision & Payout) | Actual Outcome (Decision & Payout) | Financial Variance (Target: ₹0) | Assertion Result (Pass / Fail) |
|---|---|---|---|---|---|
| TC001: Wrong Document Uploaded | Member submits two prescriptions for a consultation claim that requires a prescription and a hospital bill. | EARLY_STOP | EARLY_STOP | ₹0 | ✅ Pass |
| TC002: Unreadable Document | Member uploads a valid prescription but a blurry, unreadable photo of their pharmacy bill. | EARLY_STOP | EARLY_STOP | ₹0 | ✅ Pass |
| TC003: Documents Belong to Different Patients | The prescription is for Rajesh Kumar but the hospital bill is for a different patient, Arjun Mehta. | EARLY_STOP | EARLY_STOP | ₹0 | ✅ Pass |
| TC004: Clean Consultation â€” Full Approval | Complete, valid consultation claim with correct documents, valid member, covered treatment, within all limits. | APPROVED (₹1350) | APPROVED (₹1350.0) | ₹0.0 | ✅ Pass |
| TC005: Waiting Period â€” Diabetes | Member joined 2024-09-01. Claims for diabetes treatment on 2024-10-15, which is within the 90-day waiting period for diabetes. | REJECTED | REJECTED | ₹0 | ✅ Pass |
| TC006: Dental Partial Approval â€” Cosmetic Exclusion | Bill includes root canal treatment (covered) and teeth whitening (cosmetic, excluded). System must approve only the covered procedure. | PARTIAL (₹8000) | PARTIAL (₹8000) | ₹0 | ✅ Pass |
| TC007: MRI Without Pre-Authorization | MRI scan costing â‚¹15,000 submitted without pre-authorization. Policy requires pre-auth for MRI above â‚¹10,000. | REJECTED | REJECTED | ₹0 | ✅ Pass |
| TC008: Per-Claim Limit Exceeded | Claimed amount of â‚¹7,500 exceeds the per-claim limit of â‚¹5,000. | REJECTED | REJECTED | ₹0 | ✅ Pass |
| TC009: Fraud Signal â€” Multiple Same-Day Claims | Member EMP008 has already submitted 3 claims today before this one arrives. This is the 4th claim from the same member on the same day. | MANUAL_REVIEW | MANUAL_REVIEW | ₹0 | ✅ Pass |
| TC010: Network Hospital â€” Discount Applied | Valid claim at Apollo Hospitals, a network hospital. Network discount must be applied before co-pay. | APPROVED (₹3240) | APPROVED (₹3240.00) | ₹0.00 | ✅ Pass |
| TC011: Component Failure â€” Graceful Degradation | One component of your system fails mid-processing (simulate with the flag below). The overall pipeline must continue, produce a decision, and make the failure visible in the output with an appropriately reduced confidence score. | APPROVED | APPROVED (₹4000) | ₹0 | ✅ Pass |
| TC012: Excluded Treatment | Member claims for bariatric consultation and a diet program. Obesity treatment is explicitly excluded under the policy. | REJECTED | REJECTED | ₹0 | ✅ Pass |
