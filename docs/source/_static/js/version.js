// Add version number below logo
document.addEventListener('DOMContentLoaded', function() {
    // Get version from Sphinx configuration (available in the page)
    var version = '';

    // Try to get version from DOCUMENTATION_OPTIONS if available
    if (typeof DOCUMENTATION_OPTIONS !== 'undefined' && DOCUMENTATION_OPTIONS.VERSION) {
        version = 'v' + DOCUMENTATION_OPTIONS.VERSION;
    } else {
        // Fallback: try to extract from meta tags
        var versionMeta = document.querySelector('meta[name="version"]');
        if (versionMeta) {
            version = 'v' + versionMeta.getAttribute('content');
        } else {
            // Last fallback: try to find it in the page title or other elements
            var titleElement = document.querySelector('title');
            if (titleElement && titleElement.textContent.includes('SPINE')) {
                // If we can't find version, don't show anything rather than hardcoding
                return;
            }
        }
    }

    // Only proceed if we found a version
    if (!version || version === 'v') {
        return;
    }

    // Find the sidebar navigation container
    var sideNavSearch = document.querySelector('.wy-side-nav-search');
    if (sideNavSearch) {
        // Check if version element already exists
        var existingVersion = sideNavSearch.querySelector('.version-display');
        if (existingVersion) {
            return; // Prevent duplicates
        }

        // Create version element
        var versionElement = document.createElement('div');
        versionElement.className = 'version-display';
        versionElement.textContent = version;

        // Insert after the logo
        var logo = sideNavSearch.querySelector('a img, .icon');
        if (logo) {
            logo.parentElement.insertAdjacentElement('afterend', versionElement);
        } else {
            // Fallback: append to the container
            sideNavSearch.appendChild(versionElement);
        }
    }
});
