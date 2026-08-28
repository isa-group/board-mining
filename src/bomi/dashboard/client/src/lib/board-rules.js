/**
 * Board Design Rules Engine
 *
 * Defines what operations are allowed on board elements.
 * Rules prevent invalid designs and guide user interactions.
 */

export function createBoardRules() {
  const RulesModule = function () {
    // Check rules before operations
    this.canConnect = function (source, target) {
      // Can only create SemanticPreferences between ListBox elements
      return (
        source.type === 'board:ListBox' &&
        target.type === 'board:ListBox' &&
        source.id !== target.id
      )
    }

    this.canCreate = function (parent, shape) {
      // Can create ListBox within CardFlow, or CardFlow at root
      return parent.type === 'board:CardFlow' || parent.type === 'board:BoardDesign'
    }

    this.canMove = function (shape) {
      // Can move ListBox and CardFlow
      return shape.type === 'board:ListBox' || shape.type === 'board:CardFlow'
    }

    this.canDelete = function (element) {
      // Can delete lists and semantic preferences
      return element.type === 'board:ListBox' || element.type === 'board:SemanticPreference'
    }

    this.canResize = function (shape) {
      // Can resize CardFlow containers (dynamic sizing for contained lists)
      return shape.type === 'board:CardFlow'
    }
  }

  return RulesModule
}
