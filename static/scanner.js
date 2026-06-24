document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('sidebar-search-input');
    const feedback = document.getElementById('scanner-feedback');

    if (!searchInput) return;

    searchInput.addEventListener('keydown', async (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            const customerId = searchInput.value.trim();
            if (!customerId) return;

            // Basic validation: must be numeric digits only
            if (!/^\d+$/.test(customerId)) {
                showFeedback('Invalid Customer ID format. Numbers only.', 'error');
                return;
            }

            try {
                // Check if customer exists in the database
                const response = await fetch(`/api/customer_exists/${customerId}`);
                if (!response.ok) {
                    throw new Error('Database lookup failure');
                }
                const data = await response.json();

                if (data.exists) {
                    // Redirect to the customer profile deep-dive
                    window.location.href = `/customer/${customerId}`;
                } else {
                    showFeedback(`Customer ID ${customerId} not found.`, 'error');
                }
            } catch (err) {
                showFeedback('Error contacting RELOG engine.', 'error');
                console.error(err);
            }
        }
    });

    function showFeedback(message, type) {
        if (!feedback) return;
        feedback.innerText = message;
        feedback.className = `scanner-feedback status-msg status-${type}`;
        feedback.style.display = 'block';
        
        // Auto-dismiss the warning feedback after 3.5 seconds
        setTimeout(() => {
            feedback.style.display = 'none';
        }, 3500);
    }
});
