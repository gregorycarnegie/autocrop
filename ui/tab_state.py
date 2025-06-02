from collections.abc import Callable

from PyQt6.QtWidgets import QCheckBox, QComboBox, QDial, QPushButton, QSlider, QWidget

from line_edits import LineEditState, NumberLineEdit, PathLineEdit
from ui import utils as ut


class TabStateManager:
    """
    Manages state for UI tab widgets, including input validation,
    button states, and checkbox interactions.

    This class centralizes tab state management to reduce code
    duplication and ensure consistent behavior across tabs.
    """

    def __init__(self, parent: QWidget | None = None):
        """
        Initialize the tab state manager.

        Args:
            parent: Optional parent widget that owns this state manager
        """
        self.parent = parent
        self._checkbox_handlers: dict[QCheckBox, set[QCheckBox]] = {}
        self._button_dependencies: dict[QPushButton, set[QWidget]] = {}
        self._validation_handlers: dict[QWidget, Callable[[], bool]] = {}
        self._state_change_callback: Callable[[], None] | None = None

    def register_checkbox_exclusivity(self, checkbox: QCheckBox,
                                     exclude_checkboxes: set[QCheckBox]) -> None:
        """
        Register checkbox exclusivity relationships.
        When a checkbox is checked, it will uncheck the specified excluded checkboxes.

        Args:
            checkbox: The checkbox that will trigger the exclusion
            exclude_checkboxes: Set of checkboxes to uncheck when the main checkbox is checked
        """
        self._checkbox_handlers[checkbox] = exclude_checkboxes
        checkbox.toggled.connect(lambda checked: self._handle_checkbox_state(checkbox, checked))

    def _handle_checkbox_state(self, checkbox: QCheckBox, checked: bool) -> None:
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

    def register_validation_handler(self, widget: QWidget,
                                   handler: Callable[[], bool]) -> None:
        """
        Register a custom validation handler for a widget.

        Args:
            widget: The widget to validate
            handler: Callable that returns True if the widget is in a valid state
        """
        self._validation_handlers[widget] = handler

    def connect_widgets(self, *widgets: QWidget) -> None:
        """
        Connect widget signals to state update handlers.

        Args:
            *widgets: Widgets to connect to state update handlers
        """
        for widget in widgets:
            if isinstance(widget, NumberLineEdit | PathLineEdit):
                widget.textChanged.connect(self.update_button_states)
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self.update_button_states)
            elif isinstance(widget, QSlider | QDial):
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
        Update button enabled states based on registered dependencies and validation.
        """
        for button, dependencies in self._button_dependencies.items():
            # Check widget-based dependencies
            widgets_valid = all(self._is_widget_valid(widget) for widget in dependencies)

            # Get custom validation result, if any
            custom_valid = True
            if button in self._validation_handlers:
                custom_valid = self._validation_handlers[button]()

            # Button is enabled only if all validations pass
            final_state = widgets_valid and custom_valid

            # Only update if state has actually changed to avoid unnecessary updates
            if button.isEnabled() != final_state:
                ut.change_widget_state(final_state, button)

        if self._state_change_callback:
            self._state_change_callback()

    def _is_widget_valid(self, widget: QWidget) -> bool:
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
        if isinstance(widget, NumberLineEdit | PathLineEdit):
            return widget.state == LineEditState.VALID_INPUT and bool(widget.text())
        elif isinstance(widget, QComboBox):
            return bool(widget.currentText())
        elif isinstance(widget, QCheckBox | QSlider | QDial):
            # Checkboxes are typically considered valid regardless of state
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
