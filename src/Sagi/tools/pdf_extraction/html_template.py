html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background-color: #f0f0f0;
            margin: 0;
            padding: 20px;
        }
        
        .background-page {
            width: ffff-widthpt;
            height: ffff-heightpt;
            margin: 20px auto;
            padding: ffff-paddingpt;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            position: relative;
        }

        .column_container {
            display: flex;
            gap: 15px;
        } 
    </style>
</head>
<body>
    ffff-content
</body>
</html>
"""


def create_editable_container(index: int):
    return f"""
    <div class="editor-container">
        <!-- Control Panel -->
        <div class="control-panel">
            <button id="saveBtn-gjs{index}" class="control-btn">Save</button>
            <button id="toggleViewBtn-gjs{index}" class="control-btn">Edit Page</button>
            <button id="resetBtn-gjs{index}" class="control-btn">Reset</button>
        </div>

        <!-- GrapesJS Editor Container -->
        <div class="gjs-editor-container">
            <div id="gjs{index}"></div>
        </div>

        <!-- Style Manager Panel -->
        <div class="style-manager-panel" id="styleManagerPanel-gjs{index}">
            <div class="panel-header">
                Element Properties
            </div>
            
            <div class="panel-section">
                <h4>Add Elements</h4>
                <button class="add-element-btn" onclick="addRow('gjs{index}')">➕ Add Row</button>
                <button class="add-element-btn" onclick="addColumn('gjs{index}')">➕ Add Column</button>
                
                <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px; font-size: 12px; color: #666;">
                    <strong>How to use:</strong><br>
                    • <strong>Add Row:</strong> Click on any text element first, then click "Add Row" to copy its style<br>
                    • <strong>Add Column:</strong> Click on a column_container div (the container with multiple columns), then click "Add Column"<br>
                    • <strong>Position:</strong> The content will be appear on the top left of the page. You may move them to any place you want.
                </div>
            </div>

            <div class="panel-section">
                <h4>Style Manager</h4>
                <div id="style-manager-gjs{index}"></div>
            </div>
        </div>
    </div>
    """


def create_initialize(index: int):
    return f"""
            initializePageEditor('gjs{index}');
            setupEventListeners('gjs{index}');
    """


editable_html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ffff-title</title>
    
    <!-- GrapesJS CSS -->
    <link rel="stylesheet" href="https://unpkg.com/grapesjs/dist/css/grapes.min.css">
    
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html, body {
            background-color: #f0f0f0;
            margin: 0;
            padding: 20px;
        }

        .editor-container {
            display: flex;
            gap: 20px;
            max-width: none;
            margin: 20px auto 0 auto;
            justify-content: center;
            align-items: flex-start;
        }

        /* Control Panel */
        .control-panel {
            width: 200px;
            background: white;
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 15px;
            height: fit-content;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .control-btn {
            display: block;
            width: 100%;
            padding: 10px 15px;
            margin-bottom: 8px;
            border: none;
            border-radius: 4px;
            background: #dc3545;
            color: white;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.2s;
        }

        .control-btn:hover {
            background: #c82333;
        }

        .control-btn:last-child {
            margin-bottom: 0;
        }

        /* GrapesJS Editor Container */
        .gjs-editor-container {
            width: ffff-extra-width;
            height: ffff-height;
            background: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            position: relative;
        }

        /* Style Manager Panel */
        .style-manager-panel {
            width: 300px;
            background: white;
            border: 1px solid #ccc;
            border-radius: 5px;
            height: ffff-height;
            overflow-y: auto;
            display: none;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .style-manager-panel.show {
            display: block;
        }

        .panel-header {
            background: #f8f9fa;
            padding: 15px;
            border-bottom: 1px solid #ddd;
            font-weight: bold;
            color: #333;
        }

        .panel-section {
            padding: 15px;
            border-bottom: 1px solid #eee;
        }

        .panel-section h4 {
            margin: 0 0 10px 0;
            color: #333;
            font-size: 14px;
        }

        .add-element-btn {
            width: 100%;
            padding: 8px 12px;
            margin-bottom: 8px;
            border: none;
            border-radius: 4px;
            background: #28a745;
            color: white;
            font-size: 13px;
            cursor: pointer;
            transition: background 0.2s;
        }

        .add-element-btn:hover {
            background: #218838;
        }

        /* Auto-save indicator */
        .auto-save-indicator {
            position: fixed;
            top: 10px;
            right: 20px;
            background: #28a745;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            opacity: 0;
            transition: opacity 0.3s;
            z-index: 2000;
        }

        .auto-save-indicator.show {
            opacity: 1;
        }

        /* GrapesJS overrides */
        .gjs-cv-canvas {
            top: 0;
            width: 100% !important;
            height: 100% !important;
        }

        /* Hide default panels */
        .gjs-pn-panels {
            display: none !important;
        }

        .gjs-toolbar {
            display: none !important;
        }

        .gjs-pn-panel#gjs-pn-layers-panel { display: none !important; }
        .gjs-pn-panel#gjs-layers-container { display: none !important; }
        .gjs-pn-btn[data-id="open-layers"] { display: none !important; }

        /* Responsive */
        @media (max-width: 1024px) {
            .editor-container {
                flex-direction: column;
                align-items: center;
            }
            
            .gjs-editor-container {
                width: 100%;
                max-width: 800px;
            }
            
            .style-manager-panel {
                width: 100%;
                max-width: 800px;
                height: 400px;
            }
        }
    </style>
</head>
<body>
    <!-- Auto-save indicator -->
    <div class="auto-save-indicator" id="autoSaveIndicator">Auto-saved ✓</div>

    <div class="control-panel" style="margin: 0 auto; display: flex; justify-content: center;">
        <button id="saveAllBtn" class="control-btn" onclick="saveallHTML()">Download All Pages</button>
    </div>

    ffff-container

    <!-- GrapesJS JavaScript -->
    <script src="https://unpkg.com/grapesjs"></script>
    
    <script>
        let editors = {};
        let isEditModes = {};
        let autoSaveTimeouts = {};
        let class_dict = ffff-content

        // Function to initialize editor for a specific page (according to the class name)
        async function initializePageEditor(className) {
            const editorId = className;
            
            // Initialize state for this editor (editorId is the class name)
            editors[editorId] = null;
            isEditModes[editorId] = false;
            autoSaveTimeouts[editorId] = null;

            try {
                editors[editorId] = grapesjs.init({
                    container: `#${className}`,
                    width: 'ffff-width',
                    height: 'ffff-height',
                    storageManager: false,
                    avoidInlineStyle: false,
                    panels: { defaults: [] },
                    styleManager: {
                        appendTo: `#style-manager-${className}`,
                        sectors: [{
                            name: 'Text',
                            open: true,
                            properties: [
                                'font-size',
                                'font-family', 
                                'color',
                                'text-align',
                                'font-weight',
                                'line-height'
                            ]
                        }, {
                            name: 'Appearance',
                            open: true,
                            properties: [
                                'background-color',
                                'border',
                                'border-radius',
                                'padding',
                                'margin'
                            ]
                        }, {
                            name: 'Size & Position',
                            open: true,
                            properties: [
                                'width',
                                'height',
                                'position',
                                'top',
                                'left'
                            ]
                        }]
                    },
                    canvas: { styles: [], scripts: [] }
                });

                // Set initial content
                const contentWithStyle = `
                    <style>
                        .column_container {
                            display: flex;
                            gap: 15px;
                        }
                    </style>
                    ${class_dict[className]}
                `;
                editors[editorId].setComponents(contentWithStyle);

                editors[editorId].on('load', () => {
                    setTimeout(() => {
                        // Get all components
                        const allComponents = editors[editorId].getWrapper().find('*');
                        
                        allComponents.forEach(component => {
                            const element = component.getEl();
                            if (element && element.getAttribute('style')) {
                                const styleAttr = element.getAttribute('style');
                                const styles = {};
                                
                                // Parse existing inline styles
                                styleAttr.split(';').forEach(rule => {
                                    const [prop, val] = rule.split(':').map(s => s.trim());
                                    if (prop && val) {
                                        styles[prop] = val;
                                    }
                                });
                                
                                // Apply to GrapesJS component
                                component.setStyle(styles);
                            }
                        });
                    }, 3000);
                    
                    const canvas = editors[editorId].Canvas;
                    canvas.getFrameEl().style.pointerEvents = 'none';
                });

                // Make styles sync when changed
                editors[editorId].on('component:update:style', (component) => {
                    const element = component.getEl();
                    if (element) {
                        const styles = component.getStyle();
                        const styleString = Object.keys(styles)
                            .map(prop => `${prop}: ${styles[prop]}`)
                            .join('; ');
                        element.setAttribute('style', styleString);
                    }
                });

                setupAutoSave(editorId);
                setViewMode(editorId);

                const stylePanel = document.getElementById(`styleManagerPanel-${className}`);
                if (stylePanel) {
                    stylePanel.classList.remove('show');
                }

                console.log(`GrapesJS initialized successfully for ${className}`);
            } catch (error) {
                console.error(`Error initializing GrapesJS for ${className}:`, error);
            }
        }

        // Setup event listeners for a specific page (according to the class name)
        function setupEventListeners(className) {
            const saveBtn = document.getElementById(`saveBtn-${className}`);
            const toggleViewBtn = document.getElementById(`toggleViewBtn-${className}`);
            const resetBtn = document.getElementById(`resetBtn-${className}`);

            if (saveBtn) {
                saveBtn.addEventListener('click', () => saveHTML(className));
            }
            if (toggleViewBtn) {
                toggleViewBtn.addEventListener('click', () => toggleEditMode(className));
            }
            if (resetBtn) {
                resetBtn.addEventListener('click', () => resetContent(className));
            }
        }

        // Toggle edit mode for a specific page
        function toggleEditMode(className) {
            const editorId = className;
            isEditModes[editorId] = !isEditModes[editorId];
            const toggleBtn = document.getElementById(`toggleViewBtn-${className}`);
            const stylePanel = document.getElementById(`styleManagerPanel-${className}`);
            
            if (isEditModes[editorId]) {
                if (toggleBtn) toggleBtn.textContent = 'View Result';
                if (stylePanel) stylePanel.classList.add('show');
                setEditMode(editorId);
            } else {
                if (toggleBtn) toggleBtn.textContent = 'Edit Page';
                if (stylePanel) stylePanel.classList.remove('show');
                setViewMode(editorId);
            }
        }

        // Set edit mode for a specific page
        function setEditMode(editorId) {
            if (editors[editorId]) {
                try {
                    const canvas = editors[editorId].Canvas;
                    canvas.getFrameEl().style.pointerEvents = 'auto';
                    editors[editorId].refresh();
                } catch (error) {
                    console.error('Error setting edit mode:', error);
                }
            }
        }

        // Set view mode for a specific page
        function setViewMode(editorId) {
            if (editors[editorId]) {
                try {
                    const canvas = editors[editorId].Canvas;
                    canvas.getFrameEl().style.pointerEvents = 'none';
                    editors[editorId].select();
                } catch (error) {
                    console.error('Error setting view mode:', error);
                }
            }
        }

        // Add row function for a specific page
        function addRow(className) {
            const editorId = className;
            if (editors[editorId] && isEditModes[editorId]) {
                try {
                    // Get the currently selected component to copy its style
                    const selectedComponent = editors[editorId].getSelected();
                    let styleToCopy = '';
                    
                    if (selectedComponent) {
                
                        const component = selectedComponent.toHTML();
                        const tempDiv = document.createElement('div');
                        tempDiv.innerHTML = component;
                        const element = tempDiv.firstElementChild;
                        
                        if (element) {
                            // Extract style from the inline style attribute
                            const inlineStyle = element.getAttribute('style') || '';
                            
                            const styleObj = {};
                            inlineStyle.split(';').forEach(rule => {
                                const [property, value] = rule.split(':').map(s => s.trim());
                                if (property && value) {
                                    styleObj[property] = value;
                                }
                            });
                            
                            styleToCopy = `
                                font-family: ${styleObj['font-family'] || 'UniversLT'};
                                font-weight: ${styleObj['font-weight'] || '300'};
                                color: ${styleObj['color'] || '#000000'};
                                font-size: ${styleObj['font-size'] || '8.0pt'};
                                line-height: ${styleObj['line-height'] || '1.2'};
                                opacity: ${styleObj['opacity'] || '1.0'};
                            `;
                        }
                    }
                    
                    // If no style was copied, use default style
                    if (!styleToCopy) {
                        styleToCopy = `
                            font-family: UniversLT;
                            font-weight: 300;
                            color: #000000;
                            font-size: 8.0pt;
                            line-height: 1.2;
                            opacity: 1.0;
                        `;
                    }
                    
                    const newRow = `
                        <div style="margin-top: 10pt; padding-left: 0.0pt; overflow-wrap: break-word; ${styleToCopy} position: relative;">
                            New row content here
                        </div>
                    `;
                    editors[editorId].addComponents(newRow);
                    debouncedSave(editorId);
                } catch (error) {
                    console.error('Error adding row:', error);
                }
            }
        }

        // Add column function for a specific page
        function addColumn(className) {
            const editorId = className;
            if (editors[editorId] && isEditModes[editorId]) {
                try {
                    // Get the currently selected component
                    const selectedComponent = editors[editorId].getSelected();
                    
                    if (selectedComponent) {
                        const component = selectedComponent.toHTML();
                        const tempDiv = document.createElement('div');
                        tempDiv.innerHTML = component;
                        const element = tempDiv.firstElementChild;
                        
                        // Check if the selected element is a column_container
                        if (element && element.classList.contains('column_container')) {
                            // Add new column to the selected column_container
                            const newColumn = '<div style="width: 30pt;"><div style="margin-top: 0.0pt; padding-left: 0.0pt; overflow-wrap: break-word; line-height: 1.2; font-family: UniversLT; font-weight: 300; color: #000000; font-size: 8.0pt; opacity: 1.0;">New column content here</div></div>';
                            const newComponent = editors[editorId].addComponents(newColumn);
                            selectedComponent.append(newComponent);
                            
                        } else {
                            // Create a new column_container with 2 example columns
                            const newColumnContainer = `
                                <div class="column_container" style="margin-top: 7.0pt;">
                                    <div style="width: 60pt;">
                                        <div style="margin-top: 0.0pt; padding-left: 0.0pt; overflow-wrap: break-word; line-height: 1.2; font-family: UniversLT; font-weight: 300; color: #000000; font-size: 8.0pt; opacity: 1.0;">
                                            First column content here
                                        </div>
                                    </div>
                                    <div style="width: 60pt;">
                                        <div style="margin-top: 0.0pt; padding-left: 0.0pt; overflow-wrap: break-word; line-height: 1.2; font-family: UniversLT; font-weight: 300; color: #000000; font-size: 8.0pt; opacity: 1.0;">
                                            Second column content here
                                        </div>
                                    </div>
                                </div>
                            `;
                            editors[editorId].addComponents(newColumnContainer);
                        }
                    } else {
                        // No selection, create a new column_container with 2 example columns
                        const newColumnContainer = `
                            <div class="column_container" style="margin-top: 7.0pt;">
                                <div style="width: 60pt;">
                                    <div style="margin-top: 0.0pt; padding-left: 0.0pt; overflow-wrap: break-word; line-height: 1.2; font-family: UniversLT; font-weight: 300; color: #000000; font-size: 8.0pt; opacity: 1.0;">
                                        First column content here
                                    </div>
                                </div>
                                <div style="width: 60pt;">
                                    <div style="margin-top: 0.0pt; padding-left: 0.0pt; overflow-wrap: break-word; line-height: 1.2; font-family: UniversLT; font-weight: 300; color: #000000; font-size: 8.0pt; opacity: 1.0;">
                                        Second column content here
                                    </div>
                                </div>
                            </div>
                        `;
                        editors[editorId].addComponents(newColumnContainer);
                    }
                    
                    debouncedSave(editorId);
                } catch (error) {
                    console.error('Error adding column:', error);
                }
            }
        }

        // Save HTML for a specific page
        function saveHTML(className) {
            const editorId = className;
            if (!editors[editorId]) {
                alert('Editor not initialized');
                return;
            }

            try {
                const content = editors[editorId].getHtml();
                const css = editors[editorId].getCss();
                
                const finalHTML = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ffff-title</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background-color: #f0f0f0;
            margin: 0;
            padding: 20px;
        }
        
        .background-page {
            width: ffff-width;
            height: ffff-height;
            margin: 20px auto;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            position: relative;
        }
        ${css}
    </style>
</head>
<body>
    ${content}
</body>
</html>`;
                
                // Download file
                const blob = new Blob([finalHTML], { type: 'text/html;charset=utf-8' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `ffff-title-${className}.html`;
                a.style.display = 'none';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);

            } catch (error) {
                console.error('Save error:', error);
                alert('Error saving file. Please try again.');
            }
        }

        // Save HTML for all pages
        function saveallHTML() {
            for (const editorId in editors) {
                if (!editors[editorId]) {
                    alert('Editor not initialized');
                    return;
                }
            }

            try {
                let content = "";
                let css = "";

                for (const editorId in editors) {
                    content += editors[editorId].getHtml() + "\\n";
                    css += editors[editorId].getCss() + "\\n";
                }
                
                const finalHTML = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ffff-title</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background-color: #f0f0f0;
            margin: 0;
            padding: 20px;
        }
        
        .background-page {
            width: 597.6pt;
            height: 842.4pt;
            margin: 20px auto;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            position: relative;
        }
        ${css}
    </style>
</head>
<body>
    ${content}
</body>
</html>`;

                // Download file
                const blob = new Blob([finalHTML], { type: 'text/html;charset=utf-8' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `ffff-title.html`;
                a.style.display = 'none';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);

            } catch (error) {
                console.error('Save error:', error);
                alert('Error saving file. Please try again.');
            }
        }

        // Reset content for a specific page
        async function resetContent(className) {
            if (confirm('Reset all content? This cannot be undone.')) {
                const editorId = className;
                const initialContent = `
                    <style>
                        .column_container {
                            display: flex;
                            gap: 15px;
                        }
                    </style>
                ` + class_dict[className];
                
                if (editors[editorId]) {
                    editors[editorId].setComponents(initialContent);
                }
            }
        }

        // Auto-save functionality for a specific page
        async function saveToLocalStorage(editorId) {
            if (!isEditModes[editorId] || !editors[editorId]) return;
            
            try {
                const content = editors[editorId].getHtml();
                const css = editors[editorId].getCss();
                const data = {
                    html: content,
                    css: css,
                    timestamp: Date.now()
                };
                
                // Save to localStorage
                localStorage.setItem(`editor-${editorId}`, JSON.stringify(data));
                
                // Show auto-save indicator
                const indicator = document.getElementById('autoSaveIndicator');
                if (indicator) {
                    indicator.classList.add('show');
                    setTimeout(() => {
                        indicator.classList.remove('show');
                    }, 1000);
                }
            } catch (error) {
                console.error('Error saving to storage:', error);
            }
        }

        function debouncedSave(editorId) {
            clearTimeout(autoSaveTimeouts[editorId]);
            autoSaveTimeouts[editorId] = setTimeout(() => {
                saveToLocalStorage(editorId);
            }, 500);
        }

        function setupAutoSave(editorId) {
            if (!editors[editorId]) return;

            try {
                // Auto-save on component changes
                editors[editorId].on('component:add component:remove component:update', () => {
                    if (isEditModes[editorId]) {
                        debouncedSave(editorId);
                    }
                });

                // Auto-save on style changes
                editors[editorId].on('style:update', () => {
                    if (isEditModes[editorId]) {
                        debouncedSave(editorId);
                    }
                });
            } catch (error) {
                console.error('Error setting up auto-save:', error);
            }
        }

        // Initialize when DOM is loaded
        document.addEventListener('DOMContentLoaded', function() {
            ffff-initialize
        });
    </script>
</body>
</html>
"""
