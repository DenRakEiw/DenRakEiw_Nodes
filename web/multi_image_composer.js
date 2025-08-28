import { app } from "../../scripts/app.js";

// Register the custom node
app.registerExtension({
    name: "denrakeiw.MultiImageAspectRatioComposer",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MultiImageAspectRatioComposer") {
            
            // Store original onNodeCreated
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                // Call original onNodeCreated if it exists
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                // Add update button
                this.addWidget("button", "Update Inputs", "update", () => {
                    this.updateInputs();
                });
                
                // Initialize with current input count
                this.updateInputs();
            };
            
            // Add method to update inputs based on input_count
            nodeType.prototype.updateInputs = function() {
                const inputCountWidget = this.widgets.find(w => w.name === "input_count");
                if (!inputCountWidget) return;
                
                const inputCount = inputCountWidget.value;
                
                // Remove existing image inputs
                const imageInputs = this.inputs.filter(input => input.name.startsWith("image_"));
                imageInputs.forEach(input => {
                    this.removeInput(this.inputs.indexOf(input));
                });
                
                // Add new image inputs
                for (let i = 1; i <= inputCount; i++) {
                    this.addInput(`image_${i}`, "IMAGE");
                }
                
                // Mark for redraw
                this.setDirtyCanvas(true);
            };
            
            // Override onWidgetChanged to update inputs when input_count changes
            const onWidgetChanged = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function(widget, value, oldValue, node) {
                if (onWidgetChanged) {
                    onWidgetChanged.apply(this, arguments);
                }
                
                if (widget.name === "input_count" && value !== oldValue) {
                    this.updateInputs();
                }
            };
            
            // Add preview functionality for dimensions
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }
                
                // Display output dimensions
                if (message && message.length >= 3) {
                    const width = message[1];
                    const height = message[2];
                    
                    // Update node title to show dimensions
                    this.title = `Multi-Image Composer (${width}x${height})`;
                    
                    // Add dimension info widget if it doesn't exist
                    let dimensionWidget = this.widgets.find(w => w.name === "dimensions_info");
                    if (!dimensionWidget) {
                        dimensionWidget = this.addWidget("text", "dimensions_info", `${width}x${height}`, () => {}, {
                            serialize: false
                        });
                        dimensionWidget.disabled = true;
                    } else {
                        dimensionWidget.value = `${width}x${height}`;
                    }
                }
            };
        }
    }
});
