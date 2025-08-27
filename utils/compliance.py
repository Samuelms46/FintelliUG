import re
from typing import Dict, Any
import logging
from datetime import datetime, timedelta
import schedule
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def anonymize_text(text: str, logger: logging.Logger) -> str:
    """Anonymize PII from input text to comply with Uganda's Data Protection Act.

    Args:
        text (str): Raw text (e.g., X post).
        logger (logging.Logger): Logger for audit trails.

    Returns:
        str: Anonymized text with PII redacted.
    """
    try:
        # Basic regex patterns for common PII
        patterns = [
            (r'@[A-Za-z0-9_]+', '[REDACTED_USER]'),  # Usernames
            (r'\+256[0-9]{9}\b', '[REDACTED_PHONE]'),  # Uganda phone numbers
            (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[REDACTED_EMAIL]')  # Emails
        ]
        
        anonymized_text = text
        for pattern, replacement in patterns:
            anonymized_text = re.sub(pattern, replacement, anonymized_text)

        # Optional Presidio for advanced PII detection
        if os.getenv('USE_PRESIDIO', 'false').lower() == 'true':
            analyzer = AnalyzerEngine()
            anonymizer = AnonymizerEngine()
            results = analyzer.analyze(text=anonymized_text, entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"], language="en")
            anonymized_text = anonymizer.anonymize(text=anonymized_text, analyzer_results=results).text

        # Log anonymization action without storing original text
        log_compliance_action(
            action="anonymized_text",
            details={"text_hash": hash(text), "timestamp": datetime.now().isoformat()},
            logger=logger
        )
        return anonymized_text

    except Exception as e:
        logger.error(f"Anonymization failed: {str(e)}")
        return text  # Return original text as fallback, log for audit

def schedule_data_retention(vector_store: Any, logger: logging.Logger, retention_days: int = 90):
    """Schedule deletion of ChromaDB records older than retention period.

    Args:
        vector_store (Any): ChromaDB instance from BaseAgent.
        logger (logging.Logger): Logger for audit trails.
        retention_days (int): Retention period in days (default: 90).
    """
    def delete_old_records():
        try:
            cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
            vector_store.delete(where={"timestamp": {"$lt": cutoff}})
            log_compliance_action(
                action="deleted_records",
                details={"cutoff": cutoff, "timestamp": datetime.now().isoformat()},
                logger=logger
            )
            logger.info(f"Deleted ChromaDB records older than {cutoff}")
        except Exception as e:
            logger.error(f"Failed to delete old records: {str(e)}")

    try:
        # Schedule daily deletion
        schedule.every().day.at("02:00").do(delete_old_records)
        logger.info("Scheduled daily data retention task")
    except Exception as e:
        logger.error(f"Failed to schedule retention task: {str(e)}")

def log_compliance_action(action: str, details: Dict[str, Any], logger: logging.Logger):
    """Log compliance actions for auditability.

    Args:
        action (str): Type of compliance action (e.g., 'anonymized_text').
        details (Dict[str, Any]): Metadata for the action (e.g., text hash, timestamp).
        logger (logging.Logger): Logger for audit trails.
    """
    try:
        logger.info(f"Compliance action: {action}", extra={"details": details})
    except Exception as e:
        logger.error(f"Failed to log compliance action: {str(e)}")

def check_compliance_status(vector_store: Any, logger: logging.Logger) -> Dict[str, Any]:
    """Check compliance status for monitoring and reporting.

    Args:
        vector_store (Any): ChromaDB instance to check record timestamps.
        logger (logging.Logger): Logger for status reporting.

    Returns:
        Dict[str, Any]: Compliance metrics (e.g., oldest record date, anonymization status).
    """
    try:
        # Query ChromaDB for oldest record
        results = vector_store.similarity_search_with_score("", k=1, filter={"timestamp": {"$exists": True}})
        oldest_record_date = results[0][0].metadata.get("timestamp", None) if results else None
        
        status = {
            "oldest_record_date": oldest_record_date,
            "anonymization_enabled": os.getenv('USE_PRESIDIO', 'false').lower() == 'true',
            "retention_days": int(os.getenv('RETENTION_DAYS', 90)),
            "timestamp": datetime.now().isoformat()
        }
        
        log_compliance_action(
            action="checked_compliance_status",
            details=status,
            logger=logger
        )
        return status

    except Exception as e:
        logger.error(f"Failed to check compliance status: {str(e)}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}