# Python Date and Time Operations

This document provides a comprehensive guide to Python date, time, and datetime operations using the `datetime`, `time`, and `calendar` modules with syntax and usage examples.

## datetime Module

### datetime Objects

```python
from datetime import datetime

# Current date and time
now = datetime.now()
print(now)                                  # 2024-01-15 14:30:45.123456

# Current UTC date and time
utc_now = datetime.utcnow()
print(utc_now)                              # 2024-01-15 19:30:45.123456

# Create specific datetime
dt = datetime(2024, 1, 15, 14, 30, 45)
print(dt)                                   # 2024-01-15 14:30:45

# With microseconds
dt = datetime(2024, 1, 15, 14, 30, 45, 123456)
print(dt)                                   # 2024-01-15 14:30:45.123456

# Access components
print(dt.year)                              # 2024
print(dt.month)                             # 1
print(dt.day)                               # 15
print(dt.hour)                              # 14
print(dt.minute)                            # 30
print(dt.second)                            # 45
print(dt.microsecond)                       # 123456

# Day of week (0=Monday, 6=Sunday)
print(dt.weekday())                         # 0 (Monday)

# ISO day of week (1=Monday, 7=Sunday)
print(dt.isoweekday())                      # 1 (Monday)

# Replace components
new_dt = dt.replace(year=2025, month=3)
print(new_dt)                               # 2025-03-15 14:30:45.123456
```

### date Objects

```python
from datetime import date

# Current date
today = date.today()
print(today)                                # 2024-01-15

# Create specific date
d = date(2024, 1, 15)
print(d)                                    # 2024-01-15

# Access components
print(d.year)                               # 2024
print(d.month)                              # 1
print(d.day)                                # 15

# Day of week
print(d.weekday())                          # 0 (Monday)
print(d.isoweekday())                       # 1 (Monday)

# Replace components
new_date = d.replace(year=2025)
print(new_date)                             # 2025-01-15

# Convert from datetime
dt = datetime.now()
d = dt.date()
print(d)                                    # 2024-01-15

# Create from timestamp
d = date.fromtimestamp(1705334400)
print(d)                                    # 2024-01-15

# Create from ordinal
d = date.fromordinal(738888)
print(d)                                    # 2024-01-15
```

### time Objects

```python
from datetime import time

# Create time
t = time(14, 30, 45)
print(t)                                    # 14:30:45

# With microseconds
t = time(14, 30, 45, 123456)
print(t)                                    # 14:30:45.123456

# Access components
print(t.hour)                               # 14
print(t.minute)                             # 30
print(t.second)                             # 45
print(t.microsecond)                        # 123456

# Replace components
new_time = t.replace(hour=16)
print(new_time)                             # 16:30:45.123456

# Extract time from datetime
dt = datetime.now()
t = dt.time()
print(t)                                    # 14:30:45.123456

# Min and max time
print(time.min)                             # 00:00:00
print(time.max)                             # 23:59:59.999999
```

### timedelta Objects

```python
from datetime import datetime, timedelta

# Create timedelta
delta = timedelta(days=7)
print(delta)                                # 7 days, 0:00:00

# With various units
delta = timedelta(
    days=7,
    hours=2,
    minutes=30,
    seconds=45,
    milliseconds=500,
    microseconds=123
)
print(delta)                                # 7 days, 2:30:45.500123

# Add/subtract from datetime
now = datetime.now()
tomorrow = now + timedelta(days=1)
yesterday = now - timedelta(days=1)
next_week = now + timedelta(weeks=1)

print(tomorrow)                             # 2024-01-16 14:30:45.123456
print(yesterday)                            # 2024-01-14 14:30:45.123456
print(next_week)                            # 2024-01-22 14:30:45.123456

# Calculate difference between datetimes
dt1 = datetime(2024, 1, 15)
dt2 = datetime(2024, 1, 22)
diff = dt2 - dt1
print(diff)                                 # 7 days, 0:00:00
print(diff.days)                            # 7
print(diff.total_seconds())                 # 604800.0

# Access components
delta = timedelta(days=7, hours=2, minutes=30)
print(delta.days)                           # 7
print(delta.seconds)                        # 9000 (2h 30m in seconds)
print(delta.total_seconds())                # 612600.0

# Arithmetic with timedelta
delta1 = timedelta(days=7)
delta2 = timedelta(days=3)
print(delta1 + delta2)                      # 10 days, 0:00:00
print(delta1 - delta2)                      # 4 days, 0:00:00
print(delta1 * 2)                           # 14 days, 0:00:00
print(delta1 / 2)                           # 3 days, 12:00:00

# Negative timedelta
delta = timedelta(days=-7)
print(delta)                                # -7 days, 0:00:00
```

## Formatting and Parsing

### strftime - Format to String

```python
from datetime import datetime

dt = datetime(2024, 1, 15, 14, 30, 45)

# Common format codes
print(dt.strftime('%Y-%m-%d'))              # 2024-01-15
print(dt.strftime('%H:%M:%S'))              # 14:30:45
print(dt.strftime('%Y-%m-%d %H:%M:%S'))     # 2024-01-15 14:30:45

# Full format codes
print(dt.strftime('%A, %B %d, %Y'))         # Monday, January 15, 2024
print(dt.strftime('%a, %b %d, %Y'))         # Mon, Jan 15, 2024
print(dt.strftime('%I:%M:%S %p'))           # 02:30:45 PM
print(dt.strftime('%c'))                    # Mon Jan 15 14:30:45 2024
print(dt.strftime('%x'))                    # 01/15/24
print(dt.strftime('%X'))                    # 14:30:45

# Custom formats
print(dt.strftime('Date: %Y/%m/%d'))        # Date: 2024/01/15
print(dt.strftime('Time: %H:%M'))           # Time: 14:30

# Format codes reference:
# %Y - Year (4 digits)
# %y - Year (2 digits)
# %m - Month (01-12)
# %B - Month name (January)
# %b - Month abbr (Jan)
# %d - Day (01-31)
# %A - Weekday (Monday)
# %a - Weekday abbr (Mon)
# %H - Hour 24h (00-23)
# %I - Hour 12h (01-12)
# %M - Minute (00-59)
# %S - Second (00-59)
# %f - Microsecond
# %p - AM/PM
# %z - UTC offset
# %Z - Timezone name
# %j - Day of year (001-366)
# %U - Week number (Sunday start)
# %W - Week number (Monday start)
# %c - Locale date and time
# %x - Locale date
# %X - Locale time
# %% - Literal %
```

### strptime - Parse from String

```python
from datetime import datetime

# Parse date strings
dt = datetime.strptime('2024-01-15', '%Y-%m-%d')
print(dt)                                   # 2024-01-15 00:00:00

dt = datetime.strptime('01/15/2024', '%m/%d/%Y')
print(dt)                                   # 2024-01-15 00:00:00

# Parse datetime strings
dt = datetime.strptime('2024-01-15 14:30:45', '%Y-%m-%d %H:%M:%S')
print(dt)                                   # 2024-01-15 14:30:45

# Parse various formats
dt = datetime.strptime('Jan 15, 2024', '%b %d, %Y')
print(dt)                                   # 2024-01-15 00:00:00

dt = datetime.strptime('Monday, January 15, 2024', '%A, %B %d, %Y')
print(dt)                                   # 2024-01-15 00:00:00

# Handle parsing errors
try:
    dt = datetime.strptime('invalid date', '%Y-%m-%d')
except ValueError as e:
    print(f"Parsing error: {e}")
```

### ISO Format

```python
from datetime import datetime

dt = datetime(2024, 1, 15, 14, 30, 45, 123456)

# ISO 8601 format
iso_string = dt.isoformat()
print(iso_string)                           # 2024-01-15T14:30:45.123456

# Custom separator
iso_string = dt.isoformat(sep=' ')
print(iso_string)                           # 2024-01-15 14:30:45.123456

# Parse ISO format
dt = datetime.fromisoformat('2024-01-15T14:30:45.123456')
print(dt)                                   # 2024-01-15 14:30:45.123456

# Date ISO format
from datetime import date
d = date(2024, 1, 15)
print(d.isoformat())                        # 2024-01-15

# Time ISO format
from datetime import time
t = time(14, 30, 45)
print(t.isoformat())                        # 14:30:45
```

## Timezones

### timezone and UTC

```python
from datetime import datetime, timezone, timedelta

# UTC timezone
utc = timezone.utc
now_utc = datetime.now(utc)
print(now_utc)                              # 2024-01-15 19:30:45.123456+00:00

# Create timezone with offset
est = timezone(timedelta(hours=-5))
now_est = datetime.now(est)
print(now_est)                              # 2024-01-15 14:30:45.123456-05:00

# Convert to UTC
dt = datetime(2024, 1, 15, 14, 30, 45)
dt_utc = dt.replace(tzinfo=timezone.utc)
print(dt_utc)                               # 2024-01-15 14:30:45+00:00

# Convert between timezones
pst = timezone(timedelta(hours=-8))
est = timezone(timedelta(hours=-5))

dt_pst = datetime(2024, 1, 15, 14, 30, 45, tzinfo=pst)
dt_est = dt_pst.astimezone(est)
print(dt_pst)                               # 2024-01-15 14:30:45-08:00
print(dt_est)                               # 2024-01-15 17:30:45-05:00

# Get UTC offset
offset = dt_pst.utcoffset()
print(offset)                               # -1 day, 16:00:00

# Timezone name
print(dt_pst.tzname())                      # UTC-08:00
```

### pytz and zoneinfo (Recommended)

```python
# Using zoneinfo (Python 3.9+)
from datetime import datetime
from zoneinfo import ZoneInfo

# Create datetime with timezone
dt_ny = datetime(2024, 1, 15, 14, 30, 45, tzinfo=ZoneInfo("America/New_York"))
print(dt_ny)                                # 2024-01-15 14:30:45-05:00

# Convert between timezones
dt_la = dt_ny.astimezone(ZoneInfo("America/Los_Angeles"))
print(dt_la)                                # 2024-01-15 11:30:45-08:00

dt_tokyo = dt_ny.astimezone(ZoneInfo("Asia/Tokyo"))
print(dt_tokyo)                             # 2024-01-16 04:30:45+09:00

# Current time in specific timezone
now_ny = datetime.now(ZoneInfo("America/New_York"))
print(now_ny)

# List of common timezones:
# America/New_York, America/Chicago, America/Denver, America/Los_Angeles
# Europe/London, Europe/Paris, Europe/Berlin
# Asia/Tokyo, Asia/Shanghai, Asia/Kolkata
# Australia/Sydney, Pacific/Auckland

# Handle daylight saving time
summer = datetime(2024, 7, 15, 12, 0, tzinfo=ZoneInfo("America/New_York"))
winter = datetime(2024, 1, 15, 12, 0, tzinfo=ZoneInfo("America/New_York"))
print(summer.utcoffset())                   # -1 day, 20:00:00 (EDT, -04:00)
print(winter.utcoffset())                   # -1 day, 19:00:00 (EST, -05:00)
```

## Unix Timestamp

### Converting to/from Timestamps

```python
from datetime import datetime
import time

# Current timestamp
timestamp = time.time()
print(timestamp)                            # 1705334400.123456

# Timestamp from datetime
dt = datetime(2024, 1, 15, 14, 30, 45)
timestamp = dt.timestamp()
print(timestamp)                            # 1705334445.0

# Datetime from timestamp
dt = datetime.fromtimestamp(1705334445)
print(dt)                                   # 2024-01-15 14:30:45

# UTC datetime from timestamp
dt = datetime.utcfromtimestamp(1705334445)
print(dt)                                   # 2024-01-15 19:30:45

# Date from timestamp
from datetime import date
d = date.fromtimestamp(1705334445)
print(d)                                    # 2024-01-15

# Timestamp with microseconds
dt = datetime(2024, 1, 15, 14, 30, 45, 123456)
timestamp = dt.timestamp()
print(timestamp)                            # 1705334445.123456
```

## Calendar Operations

### calendar Module

```python
import calendar

# Check if leap year
print(calendar.isleap(2024))                # True
print(calendar.isleap(2023))                # False

# Number of leap years in range
print(calendar.leapdays(2020, 2030))        # 3

# Month calendar
cal = calendar.month(2024, 1)
print(cal)
#    January 2024
# Mo Tu We Th Fr Sa Su
#  1  2  3  4  5  6  7
#  8  9 10 11 12 13 14
# 15 16 17 18 19 20 21
# 22 23 24 25 26 27 28
# 29 30 31

# Year calendar
cal = calendar.calendar(2024)
print(cal)                                  # Full year calendar

# Month as nested list
cal = calendar.monthcalendar(2024, 1)
print(cal)
# [[1, 2, 3, 4, 5, 6, 7],
#  [8, 9, 10, 11, 12, 13, 14],
#  [15, 16, 17, 18, 19, 20, 21],
#  [22, 23, 24, 25, 26, 27, 28],
#  [29, 30, 31, 0, 0, 0, 0]]

# Month range (first weekday, number of days)
first_day, num_days = calendar.monthrange(2024, 1)
print(first_day)                            # 0 (Monday)
print(num_days)                             # 31

# Day of week
day = calendar.weekday(2024, 1, 15)
print(day)                                  # 0 (Monday)

# Month and day names
print(calendar.month_name[1])               # January
print(calendar.month_abbr[1])               # Jan
print(calendar.day_name[0])                 # Monday
print(calendar.day_abbr[0])                 # Mon

# Iterate over months
for month in range(1, 13):
    print(calendar.month_name[month])

# Text calendar with custom settings
c = calendar.TextCalendar(firstweekday=6)   # Sunday as first day
print(c.formatmonth(2024, 1))
```

### Date Ranges and Iteration

```python
from datetime import datetime, timedelta, date

# Iterate over date range
start_date = date(2024, 1, 1)
end_date = date(2024, 1, 10)
current_date = start_date

while current_date <= end_date:
    print(current_date)
    current_date += timedelta(days=1)

# Using list comprehension
date_range = [start_date + timedelta(days=x)
              for x in range((end_date - start_date).days + 1)]

# Get all Mondays in a month
import calendar

year, month = 2024, 1
cal = calendar.monthcalendar(year, month)
mondays = [week[0] for week in cal if week[0] != 0]
print(mondays)                              # [1, 8, 15, 22, 29]

# Get first Monday of month
for week in cal:
    if week[0] != 0:
        first_monday = date(year, month, week[0])
        break

print(first_monday)                         # 2024-01-01

# Last day of month
last_day = calendar.monthrange(year, month)[1]
last_date = date(year, month, last_day)
print(last_date)                            # 2024-01-31
```

## Common Operations

### Date Arithmetic

```python
from datetime import datetime, date, timedelta

# Add/subtract days
today = date.today()
tomorrow = today + timedelta(days=1)
yesterday = today - timedelta(days=1)
next_week = today + timedelta(weeks=1)
next_month = today + timedelta(days=30)    # Approximate

# Calculate age
birth_date = date(1990, 5, 15)
today = date.today()
age = (today - birth_date).days // 365
print(f"Age: {age}")

# Days until event
event_date = date(2024, 12, 25)
today = date.today()
days_until = (event_date - today).days
print(f"Days until Christmas: {days_until}")

# Working days between dates
def count_working_days(start_date, end_date):
    working_days = 0
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:           # Monday=0, Friday=4
            working_days += 1
        current += timedelta(days=1)
    return working_days

start = date(2024, 1, 1)
end = date(2024, 1, 31)
print(count_working_days(start, end))       # 23

# Get next weekday
def next_weekday(d, weekday):
    days_ahead = weekday - d.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return d + timedelta(days_ahead)

today = date.today()
next_monday = next_weekday(today, 0)        # 0 = Monday
print(next_monday)
```

### Comparisons

```python
from datetime import datetime, date

# Compare dates
date1 = date(2024, 1, 15)
date2 = date(2024, 1, 20)

print(date1 < date2)                        # True
print(date1 > date2)                        # False
print(date1 == date2)                       # False
print(date1 != date2)                       # True

# Compare datetimes
dt1 = datetime(2024, 1, 15, 10, 30)
dt2 = datetime(2024, 1, 15, 14, 30)

print(dt1 < dt2)                            # True

# Find min/max
dates = [date(2024, 1, 15), date(2024, 1, 10), date(2024, 1, 20)]
print(min(dates))                           # 2024-01-10
print(max(dates))                           # 2024-01-20

# Sort dates
dates.sort()
print(dates)                                # [2024-01-10, 2024-01-15, 2024-01-20]

# Check if date is in range
start = date(2024, 1, 1)
end = date(2024, 12, 31)
check_date = date(2024, 6, 15)

if start <= check_date <= end:
    print("Date is in 2024")
```

### Relative Dates

```python
from datetime import datetime, timedelta
import calendar

# Start/end of current month
today = datetime.today()
start_of_month = today.replace(day=1)
last_day = calendar.monthrange(today.year, today.month)[1]
end_of_month = today.replace(day=last_day)

print(start_of_month)                       # 2024-01-01 14:30:45.123456
print(end_of_month)                         # 2024-01-31 14:30:45.123456

# Start/end of current year
start_of_year = today.replace(month=1, day=1)
end_of_year = today.replace(month=12, day=31)

# Start/end of current week
start_of_week = today - timedelta(days=today.weekday())
end_of_week = start_of_week + timedelta(days=6)

# Start/end of current day
start_of_day = today.replace(hour=0, minute=0, second=0, microsecond=0)
end_of_day = today.replace(hour=23, minute=59, second=59, microsecond=999999)

# First Monday of month
def first_monday_of_month(year, month):
    cal = calendar.monthcalendar(year, month)
    for week in cal:
        if week[0] != 0:
            return date(year, month, week[0])

print(first_monday_of_month(2024, 1))       # 2024-01-01
```

## Performance and Best Practices

### Performance Tips

```python
# Use date instead of datetime when time is not needed
from datetime import date, datetime

# Faster
d = date.today()

# Slower (includes time)
dt = datetime.now()

# Reuse datetime objects
now = datetime.now()
today = now.date()
current_time = now.time()

# Use appropriate precision
# Don't keep microseconds if not needed
dt = datetime.now().replace(microsecond=0)

# Cache computed values
class Event:
    def __init__(self, event_date):
        self.event_date = event_date
        self._days_until = None

    @property
    def days_until(self):
        if self._days_until is None:
            self._days_until = (self.event_date - date.today()).days
        return self._days_until
```

### Best Practices

```python
# Always use timezone-aware datetimes for production
from datetime import datetime, timezone

# Bad - naive datetime
dt = datetime.now()

# Good - timezone-aware
dt = datetime.now(timezone.utc)

# Store in UTC, display in local time
utc_time = datetime.now(timezone.utc)
# Convert to local when displaying

# Use ISO format for serialization
dt = datetime.now(timezone.utc)
iso_string = dt.isoformat()
# Store iso_string in database/file

# Parse back
dt = datetime.fromisoformat(iso_string)

# Validate date inputs
def parse_date(date_string):
    try:
        return datetime.strptime(date_string, '%Y-%m-%d').date()
    except ValueError:
        raise ValueError("Invalid date format. Use YYYY-MM-DD")

# Use constants for common values
from datetime import timedelta

ONE_DAY = timedelta(days=1)
ONE_WEEK = timedelta(weeks=1)
ONE_HOUR = timedelta(hours=1)

tomorrow = date.today() + ONE_DAY
next_week = date.today() + ONE_WEEK
```

---

*This document covers comprehensive date and time operations in Python. For the most up-to-date information, refer to the official Python documentation.*
