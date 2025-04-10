from typing import Optional
from collections.abc import Callable
from PyQt6 import QtWidgets

from line_edits import NumberLineEdit, PathLineEdit, LineEditState
from ui import utils as ut


class TabStateManager:
    """
    Manages state for UI tab widgets, including input validation,
    button states, and checkbox interactions.
    
    This class centralizes tab state management to reduce code
    duplication and ensure consistent behavior across tabs.
    """
    
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        """
        Initialize the tab state manager.
        
        Args:
            parent: Optional parent widget that owns this state manager
        """
        self.parent = parent
        self._checkbox_handlers: dict[QtWidgets.QCheckBox, set[QtWidgets.QCheckBox]] = {}
        self._button_dependencies: dict[QtWidgets.QPushButton, set[QtWidgets.QWidget]] = {}
        self._validation_handlers: dict[QtWidgets.QWidget, Callable[[], bool]] = {}
        self._state_change_callback: Optional[Callable[[], None]] = None
        
    def register_checkbox_exclusivity(self, checkbox: QtWidgets.QCheckBox, 
                                     exclude_checkboxes: set[QtWidgets.QCheckBox]) -> None:
        """
        Register checkbox exclusivity relationships.
        When a checkbox is checked, it will uncheck the specified excluded checkboxes.
        
        Args:
            checkbox: The checkbox that will trigger the exclusion
            exclude_checkboxes: Set of checkboxes to uncheck when the main checkbox is checked
        """
        self._checkbox_handlers[checkbox] = exclude_checkboxes
        checkbox.toggled.connect(lambda checked: self._handle_checkbox_state(checkbox, checked))
        
    def _handle_checkbox_state(self, checkbox: QtWidgets.QCheckBox, checked: bool) -> None:
        """
        Handle checkbox state changes based on registered exclusivity relationships.
        
        Args:
            checkbox: The checkbox whose state changed
            checked: Whether the checkbox is now checked
        """
        if checked and checkbox in self._checkbox_handlers:
            for excluded in self._checkbox_handlers[checkbox]:
                excluded.blockSignals(True)
                excluded.setChecked(False)
                excluded.blockSignals(False)

    def register_validation_handler(self, widget: QtWidgets.QWidget,
                                   handler: Callable[[], bool]) -> None:
        """
        Register a custom validation handler for a widget.
        
        Args:
            widget: The widget to validate
            handler: Callable that returns True if the widget is in a valid state
        """
        self._validation_handlers[widget] = handler
        
    def connect_widgets(self, *widgets: QtWidgets.QWidget) -> None:
        """
        Connect widget signals to state update handlers.
        
        Args:
            *widgets: Widgets to connect to state update handlers
        """
        for widget in widgets:
            if isinstance(widget, (NumberLineEdit, PathLineEdit)):
                widget.textChanged.connect(self.update_button_states)
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.currentTextChanged.connect(self.update_button_states)
            elif isinstance(widget, (QtWidgets.QSlider, QtWidgets.QDial)):
                widget.valueChanged.connect(self.update_button_states)
    
    def set_state_change_callback(self, callback: Callable[[], None]) -> None:
        """
        Set a callback to be called whenever state changes.
        
        Args:
            callback: Function to call on state changes
        """
        self._state_change_callback = callback
        
    def update_button_states(self) -> None:
        """
        Update button enabled states based on registered dependencies and path validation.
        """
        for button, dependencies in self._button_dependencies.items():
            # Check widget-based dependencies
            widgets_valid = all(self._is_widget_valid(widget) for widget in dependencies)

            # Get custom validation result, if any
            custom_valid = True
            if button in self._validation_handlers:
                custom_valid = self._validation_handlers[button]()

            # Button is enabled only if all validations pass
            ut.change_widget_state(widgets_valid and custom_valid, button)

        if self._state_change_callback:
            self._state_change_callback()
    
    def _is_widget_valid(self, widget: QtWidgets.QWidget) -> bool:
        """
        Check if a widget is in a valid state.
        
        Args:
            widget: Widget to check
            
        Returns:
            bool: True if widget is valid, False otherwise
        """
        # Check for custom validation handler first
        if widget in self._validation_handlers:
            return self._validation_handlers[widget]()
        
        # Default validation strategies
        if isinstance(widget, (NumberLineEdit, PathLineEdit)):
            return widget.state == LineEditState.VALID_INPUT and bool(widget.text())
        elif isinstance(widget, QtWidgets.QComboBox):
            return bool(widget.currentText())
        elif isinstance(widget, QtWidgets.QCheckBox):
            # Checkboxes are typically considered valid regardless of state
            return True
        elif isinstance(widget, (QtWidgets.QSlider, QtWidgets.QDial)):
            # Sliders and dials are typically considered valid regardless of value
            return True
        
        # Default to True for unknown widget types
        return True
    
    def reset(self) -> None:
        """
        Reset the state manager, clearing all registrations.
        """
        self._checkbox_handlers.clear()
        self._button_dependencies.clear()
        self._validation_handlers.clear()
        self._state_change_callback = None
        
    def synchronize_checkboxes(self, source: QtWidgets.QCheckBox, target: QtWidgets.QCheckBox) -> None:
        """
        Synchronize the state of two checkboxes.
        
        Args:
            source: The checkbox whose state will be copied
            target: The checkbox that will match source's state
        """
        # Connect in both directions with signal blocking to prevent infinite loops
        source.toggled.connect(lambda checked: self._sync_checkbox(source, target))
        target.toggled.connect(lambda checked: self._sync_checkbox(target, source))

    @staticmethod
    def _sync_checkbox(source: QtWidgets.QCheckBox, target: QtWidgets.QCheckBox) -> None:
        """
        Sync target checkbox with source checkbox state.
        
        Args:
            source: Source checkbox
            target: Target checkbox to sync with source
        """
        target.blockSignals(True)
        target.setChecked(source.isChecked())
        target.blockSignals(False)
