
DB schema
===
participants have messages (one to many)
participants have issues (one to many)
participants have reappraisals (one to many)

PARTICIPANT#PartID, PARTICIPANT#Timestamp
PARTICIPANT#PartID, MSG#MsgID: role, content
PARTICIPANT#PartID, ISSUE#IssueType#TEXT: issue_text
PARTICIPANT#PartID, ISSUE#IssueType#NEG: neg_emo
PARTICIPANT#PartID, ISSUE#IssueType#POS: pos_emo
PARTICIPANT#PartID, REAPS#IssueType#ReapNum: reap_text, reap_rank, reap_success, reap_believability
PARTICIPANT#PartID, SURVEY#SurveyType#ItemNum: rating

PartID = prolific id
MsgID = unix timestamp
IssueType = career | relationship

Flow
===

- issue 1 (career or relationship)
- issue 2 (career or relationship: opposite of issue 1)
- survey (primals or values)
- survey (primals or values: opposite of survey 1)
- rank reap set 1
- judge reap set 1
- rank reap set 2
- judge reap set 2
- produce report on primals and values