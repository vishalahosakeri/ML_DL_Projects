from rag import get_rag
from langchain.tools import tool
from typing import Literal

@tool
def search_hr_policy(query: str) -> str:
    """Search the Novatech HR Policy Handbook.
    Use this for any question about company policies: leave rules, probation,
    notice period, working hours, hybrid work, benefits, insurance, travel,
    code of conduct, POSH, grievances, resignation and exit."""
    docs = get_rag().invoke(query)
    if not docs:
        return "No relevant content found in the HR handbook."
    return "\n\n".join(
        f"[Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}" for d in docs
    )
 
 
# Leave rules from Section 4 of the handbook
LEAVE_POLICY = {
    "earned": {"annual": 18, "max_carry_forward": 30},
    "casual": {"annual": 8, "max_carry_forward": 0},
    "sick":   {"annual": 10, "max_carry_forward": 20},
}
 
 
@tool
def leave_balance(
    leave_type: Literal["earned", "casual", "sick"],
    days_taken: int,
    carried_forward: int = 0,
) -> str:
    """Calculate remaining leave for the current year as per Novatech policy.
    leave_type: 'earned' (EL / privilege leave), 'casual' (CL) or 'sick' (SL).
    days_taken: leave days already used this year.
    carried_forward: days carried over from last year (earned and sick only)."""
    policy = LEAVE_POLICY[leave_type]
    carry = min(carried_forward, policy["max_carry_forward"])
    total = policy["annual"] + carry
    remaining = total - days_taken
 
    result = (f"{leave_type.title()} leave: {policy['annual']} annual + {carry} carried forward "
              f"= {total} days. Used: {days_taken}. ")
    if carried_forward > carry:
        result += f"(Carry forward capped at {policy['max_carry_forward']} days.) "
    if remaining < 0:
        return result + f"Exceeded by {-remaining} days; extra days may be Leave Without Pay (LWP)."
    return result + f"Remaining: {remaining} days."
 
 
@tool
def leave_encashment(monthly_basic: float, earned_leave_balance: int) -> str:
    """Calculate earned leave encashment at exit as per Novatech policy.
    Policy: up to 30 days of earned leave are encashed, calculated on basic pay.
    monthly_basic: monthly basic salary in rupees.
    earned_leave_balance: earned leave days remaining at exit."""
    days = min(earned_leave_balance, 30)
    per_day = monthly_basic / 30          # assumption: 30-day month
    amount = per_day * days
    note = " (capped at 30 days)" if earned_leave_balance > 30 else ""
    return (f"Encashable days: {days}{note}\n"
            f"Per-day basic: Rs {per_day:,.2f}\n"
            f"Encashment amount: Rs {amount:,.2f}")
 
 
@tool
def salary_after_lwp(annual_ctc: float, lwp_days: int = 0) -> str:
    """Calculate monthly salary after Leave Without Pay (LWP) deduction.
    annual_ctc: yearly salary in rupees. 1 LPA = 100000, so 12 LPA = 1200000.
    lwp_days: unpaid leave days in the month."""
    monthly = annual_ctc / 12
    per_day = monthly / 30                # assumption: 30-day month
    deduction = per_day * lwp_days
    return (f"Monthly gross: Rs {monthly:,.2f}\n"
            f"Per-day salary: Rs {per_day:,.2f}\n"
            f"LWP deduction ({lwp_days} days): Rs {deduction:,.2f}\n"
            f"Salary after LWP: Rs {monthly - deduction:,.2f}")